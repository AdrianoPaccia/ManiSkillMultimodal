import torch
import torch.nn as nn
import numpy as np
from utils import get_distance
LOG_STD_MAX = 2
LOG_STD_MIN = -5

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
    def __init__(self, env, modes):
        super().__init__()

        #modes = env.obs_modes
        obs_spaces = [env.single_observation_space_mm[env.obs_modes.index(m)] for m in modes]
        self.encoders = nn.ModuleList([
                CNN(frames, Z_DIM) if m in ['rgb', 'depth', 'segmentation'] else MLP(s.shape[0], Z_DIM) for
                s, m in zip(obs_spaces, modes)
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
        if len(z_list)==1:
            return z_list[0], None
        d = torch.stack([get_distance(zi, z_a) for zi in z_list], axis=1).unsqueeze(-1)
        z_ = torch.stack(z_list, axis=1)
        d_ = 1/d
        coeff = d_/torch.sum(d_, axis=1).unsqueeze(-1)
        z = torch.sum(coeff*z_, axis=1)
        return z, d_

    def get_representations(self, data):
        z_obs_0 = self.get_encodings(data.obs)
        z_obs_1 = self.get_encodings(data.next_obs)
        z_0, z_1 =  torch.stack(z_obs_0, dim=0).mean(dim=0),  torch.stack(z_obs_1, dim=0).mean(dim=0)
        z_act_1 = self.tf_funtion(torch.cat([z_0, data.actions.to(z_0.device)], dim=1))
        return z_obs_0, z_0, z_obs_1, z_1, z_act_1

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

