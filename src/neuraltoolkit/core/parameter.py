from .tensor import Tensor
from .device import Device

class Parameter(Tensor):
    """
    Tensor for learnable values
    
    Parameters always have gradients
    """
    def __init__(self, data):
        super().__init__(data, requires_grad=True)
        self.name = "Parameter"