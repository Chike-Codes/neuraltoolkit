from abc import ABC, abstractmethod
from neuraltoolkit.core.tensor import Tensor

class Module(ABC):
    """Parent class for all trainable objects"""

    def __call__(self, x):
        """Calls the forward method"""
        return self.forward

    @abstractmethod
    def forward(self, x) -> Tensor:
        pass

    @abstractmethod
    def parameters(self):
        """Returns a list of module parameters"""
        pass