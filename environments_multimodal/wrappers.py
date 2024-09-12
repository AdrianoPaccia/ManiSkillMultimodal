from gymnasium import spaces
import numpy as np
from environments_multimodal.noise.noise import ImageNoise
import random
import torch

class EnvMultimodalWrapper:
    def __init__(self,
                 env,
                 noise_generators=[ImageNoise(game='manipulation', noise_types=['nonoise'])],
                 **kwargs
        ):
        for item in dir(env):
            if not item.startswith('_'):
                print(f'{item}: {getattr(env, item)}')
                setattr(self, item, getattr(env, item))

        self.env = env

        #hp
        self.data_source = kwargs['sensor_data']['data_source']
        self.obs_modes = list(env.single_observation_space.spaces['sensor_data'][self.data_source].keys())

        self.noise_frequency = kwargs['noise_frequency']
        state_shape = self.get_state().shape
        self.observation_space_mm = spaces.Tuple(tuple(
            [spaces.Box(low=-np.inf, high=np.inf, shape=state_shape)] +
            [env.observation_space.spaces['sensor_data'][self.data_source][mode] for mode in self.obs_modes]
            )
        )
        self.single_observation_space_mm = spaces.Tuple(tuple(
            [spaces.Box(low=-np.inf, high=np.inf, shape=state_shape[1:])] +
            [env.single_observation_space.spaces['sensor_data'][self.data_source][mode] for mode in self.obs_modes]
            )
        )
        self.obs_modes = ['state'] + self.obs_modes
        self.store_shape = [tuple(kwargs['store_shape'][m]) for m in self.obs_modes]

        self.action_space = env.action_space
        self.single_action_space = env.single_action_space
        self.single_state_shape = state_shape[1:]
        self.noise_generators = noise_generators


    def reset_mm(self,seed=0):
        obs, info = self.env.reset(seed=seed)
        obs_ = {'state': self.get_state(),**self._filter_obs(obs)}
        return obs_, info

    def step_mm(self,a):
        obs, reward, terminated, truncated, info = self.env.step(a)

        if 'final_observation' in info:
            info['final_observation'] = self._filter_obs(info['final_observation'])

        if 'real_next_obs' in info:
            info['real_next_obs'] = self._filter_obs(info['real_next_obs'])

        obs_ = {'state': self.get_state(),**self._filter_obs(obs)}
        return obs_, reward, terminated, truncated, info

    def render_mm(self):
        return self.env.render()

    def _filter_obs(self, obs):
        obs_ = obs['sensor_data'][self.data_source]
        if random.random() < self.noise_frequency:
            mode_to_noise = random.choice(list(obs_.keys()))
            obs_[mode_to_noise] = self.noise_generators[mode_to_noise].apply_random_noise(obs_[mode_to_noise])
        return obs_

    def _get_state_dict(self):
        state_dict = self.env._env.get_state_dict()['actors']
        obs_dict = self.env._env.get_obs()
        agent_dict = obs_dict['agent']
        extra_dict = obs_dict['extra']
        obj_pose = state_dict['cube'][:,:7]
        goal_pos = state_dict['goal_region'][:, :3]
        return {**agent_dict, **extra_dict, 'obj_pos':obj_pose, 'goal_pos':goal_pos}

    def _flat_state_dict(self, x):
        return torch.cat([v for v in x.values()], dim=1).to(self.env.device)

    def get_state(self):
        state_dict = self._get_state_dict()
        return self._flat_state_dict(state_dict).to(self.env.device)


