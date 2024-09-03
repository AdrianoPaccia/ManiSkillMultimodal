import numpy as np
import random
import os
import yaml
import torch

class ConfNoise:
    def __init__(self, game:str, noise_types: list=[], frequency:float=0.0):
        self.noise_types = noise_types
        self.frequency = frequency
        self.game = game
        with open(os.path.join(os.path.dirname(__file__), f'config/{game}.yaml'), 'r') as f:
            self.config = yaml.safe_load(f)['configurations']
        self.random_x = np.load(os.path.join(os.path.dirname(__file__),f'offline_trajectories/{self.game}.npz'),
                                     allow_pickle=True)['configurations']

        if not set(noise_types).issubset(set(self.config['available_noises'])):
            raise ValueError("Noise types not supported")


    def get_observation(self, x):
        noise = random.choice(self.noise_types)
        return self.apply_noise(noise, x)

    def apply_noise(self, x, noise_type:str):
        if noise_type == 'gaussian_noise':
            x_ = self.apply_gaussian_noise(x)
        elif noise_type == 'random_obs':
            x_ = self.get_random_observation()
        elif noise_type == 'nonoise':
            x_ = x
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")
        return x_

    def apply_random_noise(self, x):
        noise_type = random.choice(list(self.config.keys()))
        return self.apply_noise(noise_type, x), noise_type

    def apply_all_noises(self,x):
        xs_ = []
        for noise_type in self.noise_types:
            xs_.append(self.apply_noise(noise_type, xs_))
        return xs_, self.noise_types

    #all implemented noises
    def apply_gaussian_noise(self, x):
        mu, std = self.config['gaussian_noise']['mu'],self.config['gaussian_noise']['std'],
        if x.dtype == np.ndarray:
            noise = np.random.normal(mu, std, size=x.shape)
        elif x.dtype == torch.tensor:
            noise = torch.normal(mu, std, size=x.shape)
        return x + noise

    def get_random_observation(self):
        i_rand = random.randint(0, self.random_x.shape[0] - 1)
        return self.random_x[i_rand]
