import torch
import torch.nn as nn
import numpy as np
from utils import get_distance
LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        h, l = env.single_action_space.high, env.single_action_space.low
        self.register_buffer("action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32))
        # will be saved in the state_dict

    def forward(self, x):
        x = self.backbone(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_eval_action(self, x):
        x = self.backbone(x)
        mean = self.fc_mean(x)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)



# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        return self.net(x)




##### MULTIMODAL

class CNN(nn.Module):
    def __init__(self, channels=2, output_dim=256):
        super(CNN, self).__init__()
        self.encoder = nn.ModuleList()
        self.encoder.append(nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=5, stride=1))
        self.encoder.append(nn.ReLU())
        self.encoder.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2))
        self.encoder.append(nn.ReLU())
        self.encoder.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2))
        self.encoder.append(nn.ReLU())
        self.encoder.append(nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, stride=2))
        self.encoder.append(nn.ReLU())
        self.encoder.append(nn.Flatten())
        self.encoder.append(nn.Linear(in_features=1568, out_features=256))
        self.encoder.append(nn.ReLU())
        self.encoder.append(nn.Linear(in_features=256, out_features=256))
        self.encoder.append(nn.ReLU())
        self.encoder.append(nn.Linear(in_features=256, out_features=output_dim))

    def forward(self, o):
        h = o
        for i, layer in enumerate(self.encoder):
            h = layer(h)
        return h

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=64):
        super(MLP, self).__init__()

        self.encoder = nn.ModuleList()
        self.encoder.append(nn.Linear(input_dim, hidden))
        self.encoder.append(nn.ReLU())
        self.encoder.append(nn.Linear(hidden, hidden))
        self.encoder.append(nn.ReLU())
        self.encoder.append(nn.Linear(hidden, hidden))
        self.encoder.append(nn.ReLU())
        self.encoder.append(nn.Linear(hidden, output_dim))

    def forward(self, x, c=None):
        if c is None:
            h = x
        else:
            h = torch.cat([x, c], -1)
        for layer in self.encoder:
            h = layer(h)
        return h

class TransitionRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden):
        super(TransitionRegressor, self).__init__()
        self.func = MLP(input_dim, output_dim, hidden)

    def forward(self, z, a):
        h = torch.cat([z, a], dim=1)
        return self.func(h)

Z_DIM = 256
frames=2
from process import image_preprocess


class ActorMultimodal(nn.Module):
    def __init__(self, env):
        super().__init__()

        modes = env.obs_modes
        self.encoders = nn.ModuleList([
                CNN(frames, Z_DIM) if m == 'rgb' or m == 'depth' or m == 'segmentation' else MLP(s.shape[0], Z_DIM) for
                s, m in zip(env.single_observation_space_mm, modes)
            ]
        )
        self.tf_funtion = MLP(Z_DIM + env.single_action_space.shape[0], Z_DIM)
        self.backbone = nn.Sequential(
            nn.Linear(Z_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        h, l = env.single_action_space.high, env.single_action_space.low
        self.register_buffer("action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32))
        # will be saved in the state_dict

    def forward(self, x):
        z_img, z_obs = self.get_encodings(x)
        z = (z_img+z_obs)/2
        h = self.backbone(z)
        mean = self.fc_mean(h)
        log_std = self.fc_logstd(h)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_encodings(self, x):
        return [encoder(o) for encoder,o in zip(self.encoders, x)]

    def get_eval_action(self, x, z_old, a):
        z_obs = self.get_encodings(x)
        z_act = self.tf_funtion(torch.cat([z_old, a], dim=1))
        z, _ =  self.aggregate_z(z_obs, z_act)
        h = self.backbone(z)
        mean = self.fc_mean(h)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, z

    def get_train_action(self, x):
        z_obs = self.get_encodings(x)
        z = torch.stack(z_obs, dim=0).mean(dim=0)
        h = self.backbone(z)
        mean = self.fc_mean(h)
        log_std = self.fc_logstd(h)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)

    def aggregate_z(self, z_list, z_a):
        '''d1 = get_distance(z_1, z_a).unsqueeze(1)
        d2 = get_distance(z_2, z_a).unsqueeze(1)
        c1 = d2 / (d1 + d2)
        c2 = d1 / (d1 + d2)
        z_0 = c1 * z_1 + c2 * z_2
        return z_0, [d1, d2]'''
        return z_a, None
        #distances = [get_distance(z_i, z_a).unsqueeze(1) for z_i in z_list]

    def get_representations(self, data):
        z_img_0, z_snd_0 = self.get_encodings(data.obs)
        z_img_1, z_snd_1 = self.get_encodings(data.next_obs)
        z_0, z_1 = (z_img_0 + z_snd_0) / 2, (z_img_1 + z_snd_1) / 2
        z_act_1 = self.tf_funtion(torch.cat([z_0, data.actions.to(z_0.device)], dim=1))
        return (z_img_0, z_snd_0), z_0, (z_img_1, z_snd_1), z_1, z_act_1

    def load_model(self, file):
        self.load_state_dict(torch.load(file))
        print(f'Model loaded from {file}')



class SoftQNetworkMultimodal(nn.Module):
    def __init__(self, input_dim, output_dim, mode):
        super().__init__()
        self.encoder = \
            CNN(frames, Z_DIM) if mode in ['rgb', 'depth', 'segmentation'] else MLP(input_dim, Z_DIM)

        self.net = nn.Sequential(
            nn.Linear(Z_DIM + output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x, a):
        z = self.encoder(x)
        x = torch.cat([z, a], 1)
        return self.net(x)

