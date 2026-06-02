import numpy as np
from math import sqrt

def he_init_norm(fan_in:int, shape:tuple, *args, **kwargs):
    std = sqrt(2 / fan_in)
    return np.random.randn(*shape) * std

def he_init_uni(fan_in:int, shape:tuple, *args, **kwargs):
    std = sqrt(6 / fan_in)
    return np.random.uniform(-std, std, size=shape)


def he_init_norm_mod(fan_in:int, shape:tuple, alpha=0.1, *args, **kwargs):
    """ Modified version for leaky ReLU. Alpha must be the same for initization and activation"""

    std = sqrt(2 / ((1 + alpha ** 2) * fan_in))
    return np.random.randn(*shape) * std

def he_init_uni_mod(fan_in:int, shape:tuple, alpha=0.1, *args, **kwargs):
    """ Modified version for leaky ReLU. Alpha must be the same for initization and activation"""

    std = sqrt(6 / ((1 + alpha ** 2) * fan_in))
    return np.random.uniform(-std, std, size=shape)