from neuraltoolkit.modules.module import Module
from ...core import tensor_ops

class Flatten(Module):
    """
    Flattens Tensors of tensors of any shape into a one dimensional vector

    ## Shapes
        Input:
            (N, ...)

        Output:
            (N, -1)
    """
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        N = x.shape[0] # number of samples
        z = tensor_ops.reshape(x, shape=(N, -1)) # flatten the data for each sample
        return z
    
    def parameters(self):
        return []
    
    def get_state(self):
        pass

    def load_state(self, state):
        pass

    @classmethod
    def from_config(cls, config):
        return cls()