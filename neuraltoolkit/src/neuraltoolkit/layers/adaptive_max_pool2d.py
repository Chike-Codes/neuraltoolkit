from .layer import Layer
from ..core import tensor_ops
from ..ops.image_processing import *
import time

class Adaptive_Max_Pool2d(Layer):
    def __init__(
            self,
            H_out:int,
            W_out:int
            ):
        
        self.H_out = H_out
        self.W_out = W_out

        self.flat_index_map = None
        self.prev_padded_shape = None

    def forward(self, x):
        N, C, H, W = x.shape
        kernel_size, stride = calc_kernel_and_stride(H, W, self.H_out, self.W_out)

        if x.shape != self.prev_padded_shape:
            self.flat_index_map = get_im2col_indices(x.shape, kernel_size, stride)
            self.prev_padded_shape = x.shape

        z = tensor_ops.max_pool2d(x, kernel_size, stride, 0, self.flat_index_map)
        return z