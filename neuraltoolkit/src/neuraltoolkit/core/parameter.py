from .tensor import Tensor
from .device import Device

class Parameter(Tensor):
    def __init__(self, data, device=Device.CPU):
        super().__init__(data, device, requires_grad=True)
        self.name = "Parameter"