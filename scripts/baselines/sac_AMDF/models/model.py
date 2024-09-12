import torch.nn as nn
import torch
import torch

from scripts.baselines.sac_AMDF.models.base_model import TransformerProcessModel, LatentAttentionGain, TransformerProcessModelAction
from scripts.baselines.sac_AMDF.models.utils import SensorModel, DecoderModel, ImgToLatentModel
from models.base_model import CNN, MLP

from scripts.baselines.sac_TWN.models import Z_DIM
from einops import rearrange, repeat

frames=2
Z_DIM = 256
class latent_model(nn.Module):
    """
    inputs -> (list_inputs, modality_list)

    for example:
    list_inputs = [img1]
    modality_list = [0]

    states -> [batch_size, ensemble, timestep, dim_x]
    """

    def __init__(
        self,
        env,
        modes,
        num_ensemble,
        win_size,
        dim_z,
        dim_a,
        dim_gt,
        sensor_len,
    ):
        super(latent_model, self).__init__()
        self.num_ensemble = num_ensemble
        self.dim_z = dim_z
        self.dim_a = dim_a
        self.dim_gt = dim_gt
        self.win_size = win_size
        self.full_modality = list(range(0, sensor_len))
        self.modes = modes

        # instantiate model
        self.process_model = TransformerProcessModelAction(
            num_ensemble=self.num_ensemble, dim_x=self.dim_z, dim_a=self.dim_a, win_size=self.win_size, dim_model=256, num_heads=8
        )

        '''self.process_model = TransformerProcessModel(
            self.num_ensemble, self.dim_z, self.win_size, 256, 8
        )'''

        #encoders
        obs_spaces = [env.single_observation_space_mm[env.obs_modes.index(m)] for m in modes]

        self.encoders = torch.nn.ModuleList(
            [
                ImgToLatentModel(input_space=s, mode=m, frames=2, num_ensemble=self.num_ensemble, output_dim=self.dim_z)
                for s, m in zip(obs_spaces, modes)
            ]
        )

        self.attention_update = LatentAttentionGain(
            self.full_modality, self.dim_z, self.dim_z, self.num_ensemble, 4
        )

        self.decoder = DecoderModel(256, mode='state', num_ensemble=self.num_ensemble, output_dim=dim_gt)



    def forward(self, inputs, states, action):
        # decompose inputs and states
        inputs_list, mod_list = inputs

        state_old = states

        # latent features from sensors
        encoded_feat = []
        for i, mod in enumerate(mod_list):
            input_idx = self.modes.index(mod)    #mod_list[mod]
            output = self.encoders[input_idx](inputs_list[i])
            encoded_feat.append(output)

        ##### prediction step #####
        action = repeat(action, "bs dim -> bs en dim", en=self.num_ensemble)
        state_pred = self.process_model(state_old, action)
        state_pred = state_pred[:, :, -1, :]
        m_A = torch.mean(state_pred, axis=1)  # m_A -> [bs, dim_z]

        ##### update step #####
        state_new, atten = self.attention_update(state_pred, encoded_feat, mod_list)
        actual_state, m_state = self.decoder(state_new)

        # condition on the latent vector
        latent_out = []
        for latent in encoded_feat:
            _, m = self.decoder(latent)
            latent_out.append(m.to(dtype=torch.float32))
        _, m = self.decoder(state_pred)
        latent_out.append(m.to(dtype=torch.float32))

        return (
            actual_state.to(dtype=torch.float32),
            m_state.to(dtype=torch.float32),
            atten.to(dtype=torch.float32),
            latent_out,
            state_new.to(dtype=torch.float32),
        )

