from .dense import Dense
from .conv2d import Conv2d
from .flatten import Flatten
from .concatenate import Concatenate
from .max_pool2d import Max_Pool2d
from .adaptive_max_pool2d import Adaptive_Max_Pool2d


def get_layer(identifier):
    identifier = identifier.lower()
    layers = {
    "dense": Dense,
    "conv": Conv2d,
    "flatten": Flatten,
    "concatenate": Concatenate
    }

    if identifier not in layers:
        raise ValueError(f"unkown layer: {identifier}")
    return layers[identifier]
