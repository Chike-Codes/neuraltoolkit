import numpy as np
from neuraltoolkit.initializers import get_initializer
from neuraltoolkit.initializers.glorot_initializer import glorot_init_uni
from neuraltoolkit.modules.module import Module
from ...core import conv2d
from ...core.parameter import Parameter
from ...ops.image_processing import split_2d_param, get_im2col_indices

class Conv2d(Module):
    """
    Standard convolutional layer

    Applies a kernel to an image output a scaled down image of feature channels

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels in the output image
        kernel_size (int or tuple): Shape of kernel i.e. (h, w). If integer then defaults to (a, a)
        stride (int or tuple): Number of pixels to skip in each direction i.e. (h, w). If integer then defaults to (a, a)
        padding (int or tuple): number of zeros to apply to the input image vertically and horizontally i.e. (h, w). If integer then defaults to (a, a)
    """

    def __init__(
            self, 
            in_channels:int,
            out_channels:int,
            kernel_size:int|tuple,
            stride:int|tuple,
            padding:int|tuple,
            initializer=glorot_init_uni()
            ):
        super().__init__()
        self._save_hparams(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            initializer=initializer.__class__.__name__
        )

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weights = None
        self.biases = None
        
        self.kernel_h, self.kernel_w = split_2d_param(kernel_size)

        self.stride = stride
        self.pad_h, self.pad_w = split_2d_param(padding)

        self.initializer = get_initializer(initializer)
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

    def forward(self, x):
        """
        Performs a forward pass through the Conv layer
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Tensor of shape (batch_size, out_channels, out_heights, out_width)
        """
        N, C, H, W = x.shape
        padded_shape = (N, C, H + self.pad_h, W + self.pad_w)

        if padded_shape != self.prev_padded_shape:
            self.flat_index_map = get_im2col_indices(padded_shape, (self.kernel_h, self.kernel_w), self.stride)
            self.prev_padded_shape = padded_shape

        z = conv2d(x, self.weights, self.stride, (self.pad_h, self.pad_w), self.flat_index_map) + self.biases
        return z