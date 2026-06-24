import numpy as np
from math import sqrt

class he_init_norm:
    def __call__(self, fan_in:int, shape:tuple, *args, **kwargs):
        """
        Initialize weights using He normal initialization.

        Args:
            fan_in: Number of input units.
            shape: Shape of the weight tensor.

        Returns:
            Initialized weight tensor.
        """
        std = sqrt(2 / fan_in)
        return np.random.randn(*shape) * std


class he_init_uni:
    def __call__(self, fan_in:int, shape:tuple, *args, **kwargs):
        """
        Initialize weights using He uniform initialization.

        Args:
            fan_in: Number of input units.
            shape: Shape of the weight tensor.

        Returns:
            Initialized weight tensor.
        """
        std = sqrt(6 / fan_in)
        return np.random.uniform(-std, std, size=shape)


def _he_init_norm_mod(fan_in:int, shape:tuple, alpha=0.1, *args, **kwargs):
    """ Modified version for leaky ReLU. Alpha must be the same for initization and activation"""

    std = sqrt(2 / ((1 + alpha ** 2) * fan_in))
    return np.random.randn(*shape) * std

def _he_init_uni_mod(fan_in:int, shape:tuple, alpha=0.1, *args, **kwargs):
    """ Modified version for leaky ReLU. Alpha must be the same for initization and activation"""

    std = sqrt(6 / ((1 + alpha ** 2) * fan_in))
    return np.random.uniform(-std, std, size=shape)