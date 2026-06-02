from .glorot_initializer import glorot_init_uni
from .he_initializer import he_init_norm, he_init_norm_mod
from ..activations import *

ACTIVATION_TO_INIT = {
    relu: he_init_norm,
    leaky_relu: he_init_norm_mod,
    sigmoid: glorot_init_uni,
    tanh: glorot_init_uni,
    linear: glorot_init_uni,
    softmax: glorot_init_uni
}