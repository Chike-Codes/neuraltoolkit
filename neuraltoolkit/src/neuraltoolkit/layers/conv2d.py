import numpy as np
from ..core import conv2d
from ..core.parameter import Parameter
from ..initializers import ACTIVATION_TO_INIT
from ..ops.image_processing import *
from .layer import Layer
from copy import deepcopy

class Conv2d(Layer):
    def __init__(
            self, 
            in_channels:int,
            out_channels:int,
            kernel_size:int|tuple,
            stride:int|tuple,
            padding:int|tuple,
            activation
            ):
        #self.input_shape = input_shape
        #self.filter_count = filters
        #self.kernel_size = kernel_size
        #self.stride = stride
        #self.padding = padding
        #self.output_shape = None
        #self.initializer = initializer
        #self.activation = activation

        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.kernel_h, self.kernel_w = split_2d_param(kernel_size)

        self.stride = stride
        self.pad_h, self.pad_w = split_2d_param(padding)

        self.activation = activation

        self.initializer = ACTIVATION_TO_INIT[activation]
        self.initialize_parameters()

        self.flat_index_map = None
        self.prev_padded_shape = None

    def initialize_parameters(self):
        weight_values = self.initializer(
            fan_in=self.in_channels * self.kernel_h * self.kernel_w,
            fan_out=self.out_channels * self.kernel_h * self.kernel_w,
            shape=(self.out_channels, self.in_channels, self.kernel_h, self.kernel_w))
        
        bias_values = np.zeros((self.out_channels, 1, 1))

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
        N, C, H, W = x.shape
        padded_shape = (N, C, H + self.pad_h, W + self.pad_w)

        if padded_shape != self.prev_padded_shape:
            self.flat_index_map = get_im2col_indices(padded_shape, (self.kernel_h, self.kernel_w), self.stride)
            self.prev_padded_shape = padded_shape

        z = conv2d(x, self.weights, self.stride, (self.pad_h, self.pad_w), self.flat_index_map) + self.biases
        return self.activation(z)