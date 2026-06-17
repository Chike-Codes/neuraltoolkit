import numpy as np

from neuraltoolkit.modules.module import Module
from ...core.parameter import Parameter
from ...initializers import ACTIVATION_TO_INIT
from copy import deepcopy

class Dense(Module):
    """
    Standard fully connected layer

    Applies the affine transformation:

    y = xW + b

    Args:
        input_shape (int): Number of layer inputs
        output_shape (int): Number of layer outputs
        activation (object): Activation function applied to each layer output
    """


    def __init__(self, input_shape:int, output_shape:int, activation):
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.initializer = ACTIVATION_TO_INIT[activation]

        self.activation = activation

        self._initialize_parameters()

    def _initialize_parameters(self):
        weight_values = self.initializer(fan_in=self.input_shape, fan_out=self.output_shape, shape=(self.output_shape, self.input_shape))
        bias_values = np.zeros((1, self.output_shape))

        self.weights = Parameter(weight_values)
        self.biases = Parameter(bias_values)

        self.weight_optimizer.build(self.weights.shape)
        self.bias_optimizer.build(self.biases.shape)
    
    def forward(self, x):
        """
        Performs a forward pass through the Dense layer
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, input_size)

        Returns:
            Tensor of shape (batch_size, output_size)
        """
        z = x @ self.weights.T + self.biases
        return self.activation(z)
    
    def parameters(self):
        return (self.weights, self.biases)