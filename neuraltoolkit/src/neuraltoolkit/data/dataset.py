import numpy as np
from ..core import Tensor
from .subset import Subset

class Dataset:
    """
    Basic container for training data with labels
    
    Args:
        x (Tensor): training data values
        y (Tensor): training data labels
    """
    def __init__(self, x=None, y=None):
        self.set_data(x, y)

    def set_data(self, x=None, y=None):
        if not isinstance(x, Tensor) or not isinstance(y, Tensor):
            raise TypeError("Both x and y must be Tensors.")
        
        if x.data.shape[0] != y.data.shape[0]:
            raise ValueError("x and y must have the same shape")

        self.x = x
        self.y = y

    def split(self, frac, shuffle:bool=True) -> Subset:
        """Returns a training a validation subset"""
        indices = np.random.permutation(self.size, )

        split = int(len(indices) * frac)

        train_indices = indices[:split]
        val_indices = indices[split:]

        train_subset = Subset(self, train_indices)
        val_subset = Subset(self, val_indices)

        return train_subset, val_subset

    @property
    def size(self):
        return self.x.shape[0]