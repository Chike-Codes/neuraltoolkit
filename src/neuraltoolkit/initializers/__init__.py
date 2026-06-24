from .glorot_initializer import *
from .he_initializer import *


def get_initializer(id):
    if callable(id):
        return id
    
    MAP = {
        glorot_init_norm.__name__:glorot_init_norm(),
        glorot_init_uni.__name__:glorot_init_uni(),
        he_init_norm.__name__:he_init_norm(),
        he_init_uni.__name__:he_init_uni()
    }

    try:
        return MAP[id]
    except:
        raise ValueError ("Initializer passed does not exist in initializer list")