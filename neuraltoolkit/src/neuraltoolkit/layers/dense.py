import numpy as np
from ..core.parameter import Parameter
from ..initializers import ACTIVATION_TO_INIT
from .layer import Layer
from copy import deepcopy

class Dense(Layer):
    
    def __init__(self, input_shape:int, output_shape:int, activation):
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.initializer = ACTIVATION_TO_INIT[activation]

        self.activation = activation

        self.initialize_parameters()

    def initialize_parameters(self):
        weight_values = self.initializer(fan_in=self.input_shape, fan_out=self.output_shape, shape=(self.output_shape, self.input_shape))
        bias_values = np.zeros((1, self.output_shape))

        self.weights = Parameter(weight_values)
        self.biases = Parameter(bias_values)
    
    def initialize_optimizers(self, optimizer):
        self.weight_optimizer = deepcopy(optimizer)
        self.bias_optimizer = deepcopy(optimizer)

        self.weight_optimizer.build(self.weights.shape)
        self.bias_optimizer.build(self.biases.shape)
    
    def optimize_parameters(self):
        self.weights.data -= self.weight_optimizer.optimize(self.weights.grad)
        self.biases.data -= self.bias_optimizer.optimize(self.biases.grad)

        self.weights.clear_grad()
        self.biases.clear_grad()
    
    def forward(self, x):
        z = x @ self.weights.T + self.biases
        return self.activation(z)