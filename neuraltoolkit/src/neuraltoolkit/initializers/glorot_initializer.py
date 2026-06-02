import numpy as np
from math import sqrt

def glorot_init_norm(fan_in:int, fan_out:int, shape:tuple):
    std = sqrt(2 / (fan_in + fan_out))
    return np.random.randn(*shape) * std

def glorot_init_uni(fan_in:int, fan_out:int, shape:tuple):
    std = sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-std, std, size=shape)