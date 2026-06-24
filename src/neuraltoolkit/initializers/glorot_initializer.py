import numpy as np
from math import sqrt

class glorot_init_norm:
    def __call__(self, fan_in:int, fan_out:int, shape:tuple):
        """
        Initialize weights using Glorot normal initialization.

        Args:
            fan_in: Number of input units.
            fan_out: Number of output units.
            shape: Shape of the weight tensor.

        Returns:
            Initialized weight tensor.
        """
        std = sqrt(2 / (fan_in + fan_out))
        return np.random.randn(*shape) * std

class glorot_init_uni:
    def __call__(self, fan_in:int, fan_out:int, shape:tuple):
        """
        Initialize weights using Glorot uniform initialization.

        Args:
            fan_in: Number of input units.
            fan_out: Number of output units.
            shape: Shape of the weight tensor.

        Returns:
            Initialized weight tensor.
        """
        std = sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-std, std, size=shape)