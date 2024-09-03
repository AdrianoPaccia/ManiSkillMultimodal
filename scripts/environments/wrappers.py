from gymnasium import spaces
import numpy as np
from scripts.environments.noise.image_noise import ImageNoise
from scripts.environments.noise.configuration_noise import ConfNoise
import random

class EnvMultimodalWrapper:
    def __init__(self, env, **kwargs):
        for item in dir(env):
            if not item.startswith('_'):
                print(f'{item}: {getattr(env, item)}')
                setattr(self, item, getattr(env, item))
        image_shape = kwargs['image_shape']
        self.observation_space_mm = spaces.Tuple((
            spaces.Box(low=0, high=255, shape=(env.num_envs, *image_shape), dtype=np.uint8),  # Image space
            env.observation_space         # Vector space
        ))
        self.single_observation_space_mm = spaces.Tuple((
            spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8),  # Image space
            env.single_observation_space         # Vector space
        ))
        self.action_space = env.action_space
        self.single_action_space = env.single_action_space
        self.env = env
        self.image_noise_generator = ImageNoise(
            game=kwargs['game'],
            frequency=kwargs['image_noise']['frequency'],
            noise_types=kwargs['image_noise']['types']
        )
        self.conf_noise_generator = ConfNoise(
            game=kwargs['game'],
            frequency=kwargs['conf_noise']['frequency'],
            noise_types=kwargs['conf_noise']['types']
        )


    def reset_mm(self,seed=0):
        obs, info = self.env.reset(seed=seed)
        img = self.env.render()
        return (img, obs), info

    def step_mm(self,a):
        obs, reward, terminated, truncated, info = self.env.step(a)
        img = self.env.render()
        if random.random() < self.image_noise_generator.frequency:
            img = self.image_noise_generator.apply_random_noise(img)
        return (img, obs), reward, terminated, truncated, info

    def render_mm(self):
        return self.env.render()
