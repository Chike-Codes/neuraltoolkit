from neuraltoolkit.modules.module import Module
from ...core import tensor_ops

class Flatten(Module):
    """
    Flattens an image input into a 1D vector
    """
    def __init__(self):
        pass

    def forward(self, x):
        """
        Performs a forward pass through the Flatten layer
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Tensor of shape (batch_size, 1D vector)
        """
        N = x.shape[0] # number of samples
        z = tensor_ops.reshape(x, shape=(N, -1)) # flatten the data for each sample
        return z
    
    def parameters(self):
        return []