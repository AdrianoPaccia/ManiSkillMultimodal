import numpy as np
from PIL import Image
import torch
from collections import deque
import matplotlib.pyplot as plt
from torchvision.transforms.v2 import Grayscale, Resize

def image_preprocess(image, resize_shape:tuple=(150,150), grayscale:bool=True, resize:bool=False):
    gs = Grayscale()
    rs = Resize(resize_shape)
    image = torch.permute(image, (0, 3, 1, 2))
    if grayscale: image=gs(image)
    if resize: image=rs(image)
    image = image/255
    return image

def process(obs, mode):
    if mode=='rgb':
        gs = Grayscale()
        return gs(torch.permute(obs, (0, 3, 1, 2))) / 255
    elif mode == 'depth':
        return obs.squeeze(-1).unsqueeze(1) / 32767
    elif mode == 'segmentation':
        return obs.squeeze(-1).unsqueeze(1) / 32767
    else:
        return obs

def process_obs_dict(obs:dict, old_obs:dict, modes:tuple, device='cpu'):
    obs_stack = [
        torch.cat([
            process(old_obs[mode], mode),
            process(obs[mode], mode)
        ], dim=1).float().to(device)
        for mode in modes
    ]
    return tuple(obs_stack)


