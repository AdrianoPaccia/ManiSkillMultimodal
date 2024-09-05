import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
import random
import matplotlib.pyplot as plt
from PIL import Image
import os
import yaml
import torch

class ImageNoise:
    def __init__(self, game:str, noise_types: list=[]):
        self.noise_types = noise_types
        self.game = game

        with open(os.path.join(os.path.dirname(__file__), f'config/{game}.yaml'), 'r') as f:
            self.config = yaml.safe_load(f)['images']

        # TODO: collect offline trajectories
        self.random_x = None

        if not set(noise_types).issubset(set(self.config['available_noises'])):
            raise ValueError("Noise types not supported")

    def get_observation(self, image):
        noise = random.choice(self.noise_types)
        return self.apply_noise(noise, image)

    def apply_noise(self,x, noise_type: str):
        if noise_type == 'gaussian_noise':
            x_ = self.apply_gaussian_noise(x)
        elif noise_type == 'background_noise':
            x_ = self.apply_background_noise(x)
        elif noise_type == 'confounders_noise':
            x_ = self.apply_confounders_noise(x)
        elif noise_type == 'random_obs':
            x_ = self.get_random_observation()
        elif noise_type == 'nonoise':
            x_ = x
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")
        return x_

    def apply_random_noise(self, x):
        noise_type = random.choice(list(self.noise_types))
        return self.apply_noise( x, noise_type)

    def apply_all_noises(self, x):
        xs_ = []
        for noise_type in self.noise_types:
            xs_.append(self.apply_noise(noise_type, x))
        return xs_, self.noise_types

    ############ NOISES

    def apply_gaussian_noise(self, x):
        mean = self.config['gaussian_noise']['mu']
        stddev = self.config['gaussian_noise']['std']
        if x.dtype == np.ndarray:
            noise = np.random.normal(mean, stddev, x.shape).astype(np.uint8)
            x_ = np.clip(x.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
        else:
            noise = torch.normal(mean, stddev, x.shape).to(x.device)
            x_ = torch.clip(x+noise, 0, 255 )

        return x_

    def apply_confounders_noise(self, xs):
        if xs.dtype == np.ndarray:
            patches = np.zeros(xs.shape)
        else:
            patches = torch.zeros(xs.shape)
        max_patch_size = self.config['confounders_noise']['max_size']
        min_patch_size = self.config['confounders_noise']['min_size']
        max_num_patches = self.config['confounders_noise']['max_num_patches']

        if len(xs.shape) == 4:
            m, height, width, _ = xs.shape
            for i in range(m):
                n = random.randint(1, max_num_patches)
                for _ in range(n):
                    side_1 = random.randint(min_patch_size, max_patch_size)
                    side_2 = random.randint(min_patch_size, max_patch_size)
                    x = random.randint(0, width - side_1)
                    y = random.randint(0, height - side_2)
                    for j in range(3):
                        patches[i, y:y + side_1, x:x + side_2, j] = random.uniform(0, 225)
                        xs[i, y:y + side_1, x:x + side_2, j] = 0.0
        else:
            pass

        x_ = torch.clip(xs+patches.to(xs.device), 0, 255)

        return x_

    def get_random_observation(self):
        i_rand = random.randint(0, self.random_x.shape[0] - 1)
        return self.random_x[i_rand]


    def apply_background_noise(self, xs):
        img_path = os.path.join(os.path.dirname(__file__), self.config['img_path'])
        img_shape = xs.shape[-3:-1]
        n = xs.shape[0] if len(xs.shape)==4 else 0

        # randomly get an image as background
        image_files = os.listdir(img_path)
        image_files = [file for file in image_files if file.endswith(('.jpg', '.jpeg', '.png'))]
        random_image = random.choice(image_files)
        bg_image = Image.open(os.path.join(img_path, random_image))
        bg_image = bg_image.resize(img_shape)
        x_bg = np.stack([np.array(bg_image)]*n)

        # background_image = np.random.randint(0, 256, original_image.shape, dtype=np.uint8)
        if xs.dtype == np.array:
            xs_np = xs
        else:
            xs_np = xs.cpu().numpy()

        floor_colors = self.config.floor_colors
        mask_bg = np.all(np.expand_dims(np.mean(xs_np, axis=-1), -1) <= [1.0], axis=-1)
        mask_up = np.all(np.expand_dims(np.mean(xs_np, axis=-1), -1) <= [floor_colors[1]], axis=-1)
        mask_low = np.all(np.expand_dims(np.mean(xs_np, axis=-1), -1) >= [floor_colors[0]], axis=-1)

        mask = mask_bg + mask_up*mask_low

        background_mask = np.stack([mask] * 3, axis=-1)
        foreground_mask = 1.0 - background_mask

        # Combine the foreground and background
        composite_image = (background_mask * x_bg) + (foreground_mask * xs_np)
        composite_image = np.clip(composite_image, 0, 255).astype(np.uint8)

        return composite_image


class DepthNoise:
    def __init__(self, game:str, noise_types: list=[]):
        self.noise_types = noise_types
        self.game = game

        with open(os.path.join(os.path.dirname(__file__), f'config/{game}.yaml'), 'r') as f:
            self.config = yaml.safe_load(f)['depth']

        #TODO: collect offline trajectories
        self.random_x = None

        if not set(noise_types).issubset(set(self.config['available_noises'])):
            raise ValueError("Noise types not supported")

    def get_observation(self, image):
        noise = random.choice(self.noise_types)
        return self.apply_noise(noise, image)

    def apply_noise(self,x, noise_type: str):
        if noise_type == 'gaussian_noise':
            x_ = self.apply_gaussian_noise(x)
        elif noise_type == 'confounders_noise':
            x_ = self.apply_confounders_noise(x)
        #elif noise_type == 'random_obs':
        #    x_ = self.get_random_observation()
        elif noise_type == 'nonoise':
            x_ = x
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")
        return x_

    def apply_random_noise(self, x):
        noise_type = random.choice(list(self.noise_types))
        return self.apply_noise( x, noise_type)

    def apply_all_noises(self, x):
        xs_ = []
        for noise_type in self.noise_types:
            xs_.append(self.apply_noise(noise_type, x))
        return xs_, self.noise_types

    ############ NOISES

    def apply_gaussian_noise(self, x):
        mean = self.config['gaussian_noise']['mu']
        stddev = self.config['gaussian_noise']['std']
        if x.dtype == np.ndarray:
            noise = np.random.normal(mean, stddev, x.shape).astype(np.uint8)
            x_ = np.clip(x.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
        else:
            noise = torch.normal(mean, stddev, x.shape).to(x.device)
            x_ = torch.clip(x+noise, 0, 255 )

        return x_

    def apply_confounders_noise(self, xs):
        if xs.dtype == np.ndarray:
            patches = np.zeros(xs.shape)
        else:
            patches = torch.zeros(xs.shape)
        max_patch_size = self.config['confounders_noise']['max_size']
        min_patch_size = self.config['confounders_noise']['min_size']
        max_num_patches = self.config['confounders_noise']['max_num_patches']

        if len(xs.shape) == 4:
            m, height, width, _ = xs.shape
            for i in range(m):
                n = random.randint(1, max_num_patches)
                for _ in range(n):
                    side_1 = random.randint(min_patch_size, max_patch_size)
                    side_2 = random.randint(min_patch_size, max_patch_size)
                    x = random.randint(0, width - side_1)
                    y = random.randint(0, height - side_2)
                    patches[i, y:y + side_1, x:x + side_2] = random.uniform(0, 225)
                    xs[i, y:y + side_1, x:x + side_2] = 0.0
        else:
            pass

        x_ = torch.clip(xs+patches.to(xs.device), 0, 255)

        return x_

    def get_random_observation(self):
        i_rand = random.randint(0, self.random_x.shape[0] - 1)
        return self.random_x[i_rand]



class SegmentNoise:
    def __init__(self, game:str, noise_types: list=[]):
        self.noise_types = noise_types
        self.game = game

        with open(os.path.join(os.path.dirname(__file__), f'config/{game}.yaml'), 'r') as f:
            self.config = yaml.safe_load(f)['segmentation']

        #TODO: collect offline trajectories
        self.random_x = None

        if not set(noise_types).issubset(set(self.config['available_noises'])):
            raise ValueError("Noise types not supported")

    def get_observation(self, image):
        noise = random.choice(self.noise_types)
        return self.apply_noise(noise, image)

    def apply_noise(self,x, noise_type: str):
        if noise_type == 'gaussian_noise':
            x_ = self.apply_gaussian_noise(x)
        elif noise_type == 'confounders_noise':
            x_ = self.apply_confounders_noise(x)
        #elif noise_type == 'random_obs':
        #    x_ = self.get_random_observation()
        elif noise_type == 'nonoise':
            x_ = x
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")
        return x_

    def apply_random_noise(self, x):
        noise_type = random.choice(list(self.noise_types))
        return self.apply_noise( x, noise_type)

    def apply_all_noises(self, x):
        xs_ = []
        for noise_type in self.noise_types:
            xs_.append(self.apply_noise(noise_type, x))
        return xs_, self.noise_types

    ############ NOISES

    def apply_gaussian_noise(self, x):
        mean = self.config['gaussian_noise']['mu']
        stddev = self.config['gaussian_noise']['std']
        if x.dtype == np.ndarray:
            noise = np.random.normal(mean, stddev, x.shape).astype(np.uint8)
            x_ = np.clip(x.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
        else:
            noise = torch.normal(mean, stddev, x.shape).to(x.device)
            x_ = torch.clip(x+noise, 0, 255 )

        return x_

    def apply_confounders_noise(self, xs):
        if xs.dtype == np.ndarray:
            patches = np.zeros(xs.shape)
        else:
            patches = torch.zeros(xs.shape)
        max_patch_size = self.config['confounders_noise']['max_size']
        min_patch_size = self.config['confounders_noise']['min_size']
        max_num_patches = self.config['confounders_noise']['max_num_patches']

        if len(xs.shape) == 4:
            m, height, width, _ = xs.shape
            for i in range(m):
                n = random.randint(1, max_num_patches)
                for _ in range(n):
                    side_1 = random.randint(min_patch_size, max_patch_size)
                    side_2 = random.randint(min_patch_size, max_patch_size)
                    x = random.randint(0, width - side_1)
                    y = random.randint(0, height - side_2)
                    patches[i, y:y + side_1, x:x + side_2] = random.uniform(0, 225)
                    xs[i, y:y + side_1, x:x + side_2] = 0.0
        else:
            pass

        x_ = torch.clip(xs+patches.to(xs.device), 0, 255)

        return x_

    def get_random_observation(self):
        i_rand = random.randint(0, self.random_x.shape[0] - 1)
        return self.random_x[i_rand]



class ConfNoise:
    def __init__(self, game:str, noise_types: list=[]):
        self.noise_types = noise_types
        self.game = game
        with open(os.path.join(os.path.dirname(__file__), f'config/{game}.yaml'), 'r') as f:
            self.config = yaml.safe_load(f)['configurations']

        #TODO: collect offline trajectories
        self.random_x = None

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


