from gymnasium import spaces
import numpy as np

image_process_shape = (150,150)
frames = 2
image_shape = (frames, *image_process_shape)

class EnvMultimodalWrapper:
    def __init__(self, env):
        for item in dir(env):
            if not item.startswith('_'):
                print(f'{item}: {getattr(env, item)}')
                setattr(self, item, getattr(env, item))
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


    def reset_mm(self,seed=0):
        obs, info = self.env.reset(seed=seed)
        img = self.env.render()
        return (img, obs), info

    def step_mm(self,a):
        obs, reward, terminated, truncated, info = self.env.step(a)
        img = self.env.render()
        return (img, obs), reward, terminated, truncated, info

    def render_mm(self):
        return self.env.render()
