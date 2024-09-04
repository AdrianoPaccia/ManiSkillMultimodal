from gymnasium import spaces
import numpy as np
from scripts.environments.noise.noise import ImageNoise
import random

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


        #hp
        self.data_source = kwargs['sensor_data']['data_source']
        self.obs_modes = list(env.single_observation_space.spaces['sensor_data'][self.data_source].keys())

        self.noise_frequency = kwargs['noise_frequency']

        self.observation_space_mm = spaces.Tuple(tuple(
            [env.observation_space.spaces['sensor_data'][self.data_source][mode] for mode in self.obs_modes]
            )
        )

        self.single_observation_space_mm = spaces.Tuple(tuple(
            [env.single_observation_space.spaces['sensor_data'][self.data_source][mode] for mode in self.obs_modes]
            )
        )
        '''self.observation_space_mm = spaces.Tuple((
            spaces.Box(low=0, high=255, shape=(env.num_envs, (2,128,128)), dtype=np.uint8),  # Image space
            env.observation_space         # Vector space
        ))
        self.single_observation_space_mm = spaces.Tuple((
            spaces.Box(low=0, high=255, shape=(2,128,128), dtype=np.uint8),  # Image space
            env.single_observation_space         # Vector space
        ))'''

        self.action_space = env.action_space
        self.single_action_space = env.single_action_space
        self.env = env

        self.noise_generators = noise_generators


    def reset_mm(self,seed=0):
        obs, info = self.env.reset(seed=seed)
        return obs['sensor_data'][self.data_source], info

    def step_mm(self,a):
        obs, reward, terminated, truncated, info = self.env.step(a)
        obs_ = obs['sensor_data'][self.data_source]
        if random.random() < self.noise_frequency:
            mode_to_noise = random.choice(self.obs_modes)
            obs_[mode_to_noise] = self.noise_generator[mode_to_noise].apply_random_noise(obs_[mode_to_noise])
        return obs_, reward, terminated, truncated, info

    def render_mm(self):
        return self.env.render()
