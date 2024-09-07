import torch.nn as nn
import torch
import torch.nn.functional as F
from bayesian_torch.layers.flipout_layers.linear_flipout import LinearFlipout
from einops import rearrange, repeat
from scripts.baselines.sac_AMDF.models.attention import PositionalEncoder, ResidualAttentionBlock, KalmanAttentionBlock



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



class TransformerProcessModel(nn.Module):
    """
    process model takes a state or a stack of states (t-n:t-1) and
    predict the next state t. the process model is flexiable, we can inject the known
    dynamics into it, we can also change the model architecture which takes sequential
    data as input

    input -> [batch_size, ensemble, timestep, dim_x]
    output ->  [batch_size, ensemble, timestep, dim_x]
    """

    def __init__(self, num_ensemble, dim_x, win_size, dim_model, num_heads):
        super(TransformerProcessModel, self).__init__()
        self.num_ensemble = num_ensemble
        self.dim_x = dim_x
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.win_size = win_size

        self.positional_encoding_layer = PositionalEncoder(
            d_model=dim_model, dropout=0.1, max_seq_len=2000, batch_first=True
        )
        self.attention_layer_1 = ResidualAttentionBlock(
            d_model=dim_model, n_head=num_heads, attn_mask=None
        )
        self.attention_layer_2 = ResidualAttentionBlock(
            d_model=dim_model, n_head=num_heads, attn_mask=None
        )
        self.attention_layer_3 = ResidualAttentionBlock(
            d_model=dim_model, n_head=num_heads, attn_mask=None
        )
        self.bayes1 = LinearFlipout(in_features=self.dim_x, out_features=32)
        self.bayes2 = LinearFlipout(in_features=32, out_features=128)
        self.bayes3 = LinearFlipout(in_features=128, out_features=256)
        self.bayes_m1 = torch.nn.Linear(256, 128)
        self.bayes_m2 = torch.nn.Linear(128, self.dim_x)

    def forward(self, input):
        batch_size = input.shape[0]
        input = rearrange(input, "n en k dim -> (n en) k dim")
        input = rearrange(input, "n k dim -> (n k) dim")

        # branch of the state
        x, _ = self.bayes1(input)
        x = F.relu(x)
        x, _ = self.bayes2(x)
        x = F.relu(x)
        x, _ = self.bayes3(x)
        x = F.relu(x)
        x = rearrange(x, "(n k) dim -> n k dim", k=self.win_size)

        # for pos embedding layers
        x = rearrange(x, "n k dim -> k n dim")
        x = self.positional_encoding_layer(x)

        x, _ = self.attention_layer_1(x)
        x, _ = self.attention_layer_2(x)
        x, _ = self.attention_layer_3(x)

        #x = rearrange(x, "k n dim -> n k dim", n=batch_size * self.num_ensemble)
        #x = rearrange(x, "n k dim -> (n k) dim", n=batch_size * self.num_ensemble)
        x = rearrange(x, "k n dim -> (n k) dim", n=batch_size * self.num_ensemble)
        x = self.bayes_m1(x)
        x = F.relu(x)
        x = self.bayes_m2(x)
        output = rearrange(x, "(n k) dim -> n k dim", n=batch_size * self.num_ensemble)
        output = rearrange(output, "(n en) k dim -> n en k dim", en=self.num_ensemble)

        return output




class LatentAttentionGain(nn.Module):
    def __init__(self, full_mod, dim_x, dim_z, dim_model, num_heads):
        """
        attention gain module is used to replace the Kalman update step, this module
        takes a predicted state and a learned observation
        learn the measuremenmt update by calculating attention of the state w.r.t to
        the concatnated vector [state,observation], and update state accordingly

        input:
        obs -> [y1, y2, y3]
        modality_list -> [0, 1, 2]
        state -> x
        y -> [batch_size, ensemble, dim_z]
        x -> [batch_size, ensemble, dim_x]


        output:
        atten -> [batch_size, dim_x, dim_y]
        """
        super(LatentAttentionGain, self).__init__()
        self.full_mod = full_mod
        self.dim_z = dim_z
        self.dim_x = dim_x
        self.num_ensemble = dim_model

        self.Q = nn.Parameter(torch.rand(self.dim_x, dim_model))

        self.positional_encoding_layer = PositionalEncoder(
            d_model=dim_model, dropout=0.1, max_seq_len=2000, batch_first=True
        )
        self.attention_layer_1 = KalmanAttentionBlock(
            d_model=dim_model, n_head=num_heads
        )

    def generate_attn_mask(self, dim_x, dim_z, full_mod, mod_list):
        tmp = torch.tensor(float("-inf"), dtype=torch.float32)
        zero = torch.tensor(0, dtype=torch.float32)
        attn_mask = torch.eye(dim_x)
        for modality in full_mod:
            if modality in mod_list:
                p2 = torch.eye(dim_z)
            else:
                p2 = torch.zeros((dim_z, dim_z))
            fill = torch.zeros((dim_x - dim_z, dim_z))
            p2 = torch.cat((fill, p2), axis=0)
            attn_mask = torch.cat((attn_mask, p2), axis=1)
        attn_mask = torch.where(attn_mask < 1, tmp, zero)
        return attn_mask

    def forward(self, state, obs_list, mod_list):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = state.shape[0]

        # generate attn_mask from

        attn_mask = self.generate_attn_mask(
            self.dim_x, self.dim_z, self.full_mod, mod_list
        )

        # zero means
        m_x = torch.mean(state, axis=1)
        m_x = repeat(m_x, "bs dim -> bs k dim", k=self.num_ensemble)
        A = state - m_x

        # collect latent obs if one modality is missing, then use 0
        Y_list = []
        Y_ = []
        idx = 0
        for modality in self.full_mod:
            if modality in mod_list:
                obs = obs_list[idx]
                m_y = torch.mean(obs, axis=1)
                m_y = repeat(m_y, "bs dim -> bs k dim", k=self.num_ensemble)
                Y = obs - m_y
                Y = rearrange(Y, "bs en dim -> dim bs en")
                Y_list.append(Y.to(device))
                Y_.append(obs)
                idx = idx + 1
            else:
                Y = torch.rand(self.dim_z, batch_size, self.num_ensemble) * 0.0
                Y_list.append(Y.to(device))
                tmp = torch.rand(batch_size, self.num_ensemble, self.dim_z) * 0.0
                Y_.append(tmp.to(device))

        # define Q
        query = repeat(self.Q, "dim en -> dim bs en", bs=batch_size)
        query = self.positional_encoding_layer(query)

        # define K
        A = rearrange(A, "bs en dim -> dim bs en")
        xy = A
        for Y in Y_list:
            xy = torch.cat((xy, Y), axis=0)
        xy = self.positional_encoding_layer(xy)
        _, atten = self.attention_layer_1(query, xy, attn_mask)

        # define V
        merge = state
        for obs in Y_:
            merge = torch.cat((merge, obs), axis=2)
        merge = rearrange(merge, "bs en dim -> bs dim en")

        # use the attention to do a weighted sum
        state = torch.matmul(atten, merge)
        state = rearrange(state, "bs dim en -> bs en dim")

        return state, atten