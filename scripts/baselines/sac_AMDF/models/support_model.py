from bayesian_torch.layers.flipout_layers.linear_flipout import LinearFlipout
import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange, repeat
from models.base_model import CNN, MLP


Z_DIM=256
frames=2
class AuxiliaryStateModel(nn.Module):
    """
    input -> [batch_size, timestep, dim_gt]
    output ->  [batch_size, ensemble, timestep, dim_x]
    """

    def __init__(self, input_dim, output_dim=Z_DIM, mode='rgb', num_ensemble=8):
        super(AuxiliaryStateModel, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.num_ensemble = num_ensemble
        self.mode = mode

        self.encoder = \
            CNN(frames, 256) if mode in ['rgb', 'depth', 'segmentation'] else MLP(input_dim, 256)

        self.fc1 = LinearFlipout(256, 512)
        self.fc2 = LinearFlipout(512, 1024)
        self.fc3 = LinearFlipout(1024, self.output_dim)


    def forward(self, x):
        batch_size = x.shape[0]

        if self.mode in ['rgb', 'depth', 'segmentation']:
            x = repeat(x, "bs k f h w -> bs en k f h w", en=self.num_ensemble)
            x = rearrange(x, "n en k f h w -> (n en) k f h w")
            x = rearrange(x, "n k f h w -> (n k) f h w")

        else:
            x = repeat(x, "bs k dim -> bs en k dim", en=self.num_ensemble)
            x = rearrange(x, "n en k dim -> (n en) k dim")
            x = rearrange(x, "n k dim -> (n k) dim")

        z = self.encoder(x)
        h, _ = self.fc1(z)
        h = F.leaky_relu(h)
        h, _ = self.fc2(h)
        h = F.leaky_relu(h)
        h, _ = self.fc3(h)

        output = rearrange(
            h, "(n k) dim -> n k dim", n=batch_size * self.num_ensemble
        )
        output = rearrange(output, "(n en) k dim -> n en k dim", en=self.num_ensemble)

        return output