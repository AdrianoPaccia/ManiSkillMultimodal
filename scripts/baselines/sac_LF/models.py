import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

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



class CrossAttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(CrossAttentionLayer, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.scale = input_dim

    def forward(self, x1, x2):
        q, k, v = self.query(x1), self.key(x2), self.value(x2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_weights, v)


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        score = torch.matmul(queries, keys.transpose(-2, -1)) / (self.input_dim ** 0.5)
        attention = self.softmax(score)
        weighted = torch.matmul(attention, values)
        return weighted

Z_DIM = 256
frames=2
available_fusions = ['linear_combination', 'concatenation', 'attention']

class ActorMultimodal(nn.Module):
    def __init__(self, env, modes, fusion_strategy:str = 'linear_combination'):
        super().__init__()

        self.modes = modes
        n = len(self.modes)

        if fusion_strategy == 'concatenation':
            z1_dim = Z_DIM//n #encoders output dimension
            z2_dim = z1_dim*n #backbone input dimension

        elif fusion_strategy == 'linear_combination':
            z1_dim, z2_dim = Z_DIM, Z_DIM
            self.comb_params = torch.nn.parameter.Parameter(torch.rand(len(self.modes), z2_dim), requires_grad=True)

        elif fusion_strategy == 'attention':
            z1_dim, z2_dim = Z_DIM, Z_DIM
            self.att_layer = SelfAttention(Z_DIM*n)
            #self.attention = CrossAttentionLayer(z_dim)

        else:
            raise ValueError(f'fusion_strategy {fusion_strategy} is not one of the available ones ({available_fusions})')
        self.fusion_strategy = fusion_strategy

        obs_spaces = [env.single_observation_space_mm[env.obs_modes.index(m)] for m in modes]
        self.encoders = nn.ModuleList([
                CNN(frames, Z_DIM) if m in ['rgb', 'depth', 'segmentation'] else MLP(s.shape[0], Z_DIM) for
                s, m in zip(obs_spaces, modes)
            ]
        )

        self.backbone = nn.Sequential(
            nn.Linear(z2_dim, 256),
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
        z_obs = self.get_encodings(x)
        z = self.aggregate(z_obs)
        h = self.backbone(z)
        mean = self.fc_mean(h)
        log_std = self.fc_logstd(h)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_encodings(self, x):
        return [encoder(o) for encoder,o in zip(self.encoders, x)]

    def get_eval_action(self, x):
        z_obs = self.get_encodings(x)
        z =  self.aggregate(z_obs)
        h = self.backbone(z)
        mean = self.fc_mean(h)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action

    def get_train_action(self, x):
        z_obs = self.get_encodings(x)
        z =  self.aggregate(z_obs)
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

    def aggregate(self, z_obs):
        n = len(z_obs)
        if n==1:
            return z_obs[0]
        if self.fusion_strategy == 'concatenation':
            return torch.cat(z_obs, dim=-1)
        elif self.fusion_strategy == 'linear_combination':
            try:
                z_ = torch.stack(z_obs, dim=1)*self.comb_params
            except:
                dio=0
            return torch.sum(z_, dim=1)
        elif self.fusion_strategy == 'attention':
            z_ = self.att_layer(torch.cat(z_obs, dim=1)).reshape(-1, n, z_obs[0].shape[-1])
            return torch.sum(z_, dim=1)
        else:
            raise ValueError('Fusion strategy is not valid!')

    def get_representations(self, obs, obs_next):
        z_obs_0 = self.get_encodings(obs)
        z_obs_1 = self.get_encodings(obs)
        z_0, z_1 =  self.aggregate(z_obs_0),  self.aggregate(z_obs_1)
        return z_obs_0, z_0, z_obs_1, z_1

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

