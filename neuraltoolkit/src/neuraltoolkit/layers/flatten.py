from ..core import tensor_ops
from .layer import Layer

class Flatten(Layer):
    def __init__(self):
        pass

    def forward(self, x):
        N = x.shape[0] # number of samples
        z = tensor_ops.reshape(x, shape=(N, -1)) # flatten the data for each sample
        return z