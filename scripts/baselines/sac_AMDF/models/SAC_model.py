import torch
import torch.nn as nn
import numpy as np
from utils import get_distance
from scripts.baselines.sac_AMDF.models.base_model import CNN, MLP, TransitionRegressor


LOG_STD_MAX = 2
LOG_STD_MIN = -5
Z_DIM = 256
frames=2


class Actor(nn.Module):
    def __init__(self, input_dim, env):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256),
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

    def forward(self, z):
        h = self.backbone(z)
        mean = self.fc_mean(h)
        log_std = self.fc_logstd(h)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats
        return mean, log_std

    def get_eval_action(self, z):
        h = self.backbone(z)
        mean = self.fc_mean(h)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action

    def get_train_action(self, z):
        h = self.backbone(z)
        mean = self.fc_mean(h)
        log_std = self.fc_logstd(h)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        std = log_std.exp()
        try:
            normal = torch.distributions.Normal(mean, std)
        except:
            dio = 0

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



class SoftQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, z, a):
        x = torch.cat([z, a], 1)
        return self.net(x)

