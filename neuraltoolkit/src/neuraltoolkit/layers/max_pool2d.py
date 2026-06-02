from .layer import Layer
from ..core import tensor_ops
from ..ops.image_processing import get_im2col_indices, split_2d_param

class Max_Pool2d(Layer):
    def __init__(
            self, 
            kernel_size:int|tuple,
            stride:int|tuple,
            padding:int|tuple
            ):
        
        self.kernel_size = kernel_size
        self.kernel_h, self.kernel_w = split_2d_param(kernel_size)
        self.stride = stride
        self.padding = padding
        self.pad_h, self.pad_w = split_2d_param(padding)

        self.flat_index_map = None
        self.prev_padded_shape = None

    def forward(self, x):
        N, C, H, W = x.shape
        padded_shape = (N, C, H + self.pad_h, W + self.pad_w)

        if padded_shape != self.prev_padded_shape:
            self.flat_index_map = get_im2col_indices(padded_shape, (self.kernel_h, self.kernel_w), self.stride)
            self.prev_padded_shape = padded_shape

        z = tensor_ops.max_pool2d(x, self.kernel_size, self.stride, self.padding, self.flat_index_map)
        return z