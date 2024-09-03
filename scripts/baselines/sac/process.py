import numpy as np
from PIL import Image
import torch
from collections import deque
import matplotlib.pyplot as plt
from torchvision.transforms.v2 import Grayscale, Resize

def image_preprocess(image, resize_shape:tuple=(150,150), grayscale:bool=True, resize:bool=True):
    gs = Grayscale()
    rs = Resize(resize_shape)
    image = torch.permute(image, (0, 3, 1, 2))
    if grayscale: image=gs(image)
    if resize: image=rs(image)
    image = image/255
    return image

def sound_preprocess(sound_tuples):
    preprocessed_sound_obs = []
    for sound_tuple in sound_tuples:
        frequency, amplitude = sound_tuple
        preprocessed_frequency = frequency / 500.0 #441.0  # Normalize frequency
        preprocessed_amplitude = amplitude / 1 #255.0  # Normalize amplitude
        preprocessed_sound_obs.extend([preprocessed_frequency, preprocessed_amplitude])
    sound = np.array(preprocessed_sound_obs)
    return sound.tolist()