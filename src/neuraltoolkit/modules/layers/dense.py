from mimetypes import init

import numpy as np
from neuraltoolkit.initializers import get_initializer
from neuraltoolkit.initializers.glorot_initializer import glorot_init_uni
from neuraltoolkit.modules.module import Module
from ...core.parameter import Parameter

class Dense(Module):
    """
    A fully connected layer.

    Applies the affine transformation

        y = xW + b

    to the input tensor.

    Args:
        in_shape:
            Number of input features.

        out_shape:
            Number of output features.

    ## Shapes
        Input:
            (N, in_features)

        Output:
            (N, out_features)

    Example:
        layer = ntk.Dense(784, 128)
        
        y = layer(x)
    """

    def __init__(self, in_shape:int, out_shape:int, initializer=glorot_init_uni()):
        super().__init__()
        self._save_hparams(
            in_shape=in_shape,
            out_shape=out_shape,
            initializer=initializer.__class__.__name__
        )

        self.in_shape = in_shape
        self.out_shape = out_shape
        
        self.weights = None
        self.biases = None

        self.initializer = get_initializer(initializer)
        self._initialize_parameters()

    def _initialize_parameters(self):
        weight_values = self.initializer(fan_in=self.in_shape, fan_out=self.out_shape, shape=(self.out_shape, self.in_shape))
        bias_values = np.zeros((1, self.out_shape))

        self.weights = Parameter(weight_values)
        self.biases = Parameter(bias_values)
    
    def forward(self, x):
        z = x @ self.weights.T + self.biases
        return z

    
