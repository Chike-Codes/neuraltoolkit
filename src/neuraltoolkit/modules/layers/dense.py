from mimetypes import init

import numpy as np
from neuraltoolkit.initializers import get_initializer
from neuraltoolkit.initializers.glorot_initializer import glorot_init_uni
from neuraltoolkit.modules.module import Module
from ...core.parameter import Parameter

class Dense(Module):
    """
    Standard fully connected layer

    Applies the affine transformation:

    y = xW + b

    Args:
        input_shape (int): Number of layer inputs
        output_shape (int): Number of layer outputs
    """

    def __init__(self, input_shape:int, output_shape:int, initializer=glorot_init_uni()):
        super().__init__()
        self._save_hparams(
            input_shape=input_shape,
            output_shape=output_shape,
            initializer=initializer.__class__.__name__
        )

        self.input_shape = input_shape
        self.output_shape = output_shape
        
        self.weights = None
        self.biases = None

        self.initializer = get_initializer(initializer)
        self._initialize_parameters()

    def _initialize_parameters(self):
        weight_values = self.initializer(fan_in=self.input_shape, fan_out=self.output_shape, shape=(self.output_shape, self.input_shape))
        bias_values = np.zeros((1, self.output_shape))

        self.weights = Parameter(weight_values)
        self.biases = Parameter(bias_values)
    
    def forward(self, x):
        """
        Performs a forward pass through the Dense layer
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, input_size)

        Returns:
            Tensor of shape (batch_size, output_size)
        """
        z = x @ self.weights.T + self.biases
        return z

    
