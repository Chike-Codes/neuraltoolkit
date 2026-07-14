from neuraltoolkit.modules.module import Module
from ...core import tensor_ops
from ...ops.image_processing import get_im2col_indices, split_2d_param

class Max_Pool2d(Module):
    """
    Scales down an image by selecting the maximum pixel values

    Args:
        kernel_size (int or tuple): 
            Shape of kernel (h, w). If integer then defaults to (a, a)

        stride (int or tuple): 
            Number of pixels to skip in each direction (h, w). If integer then defaults to (a, a)

        padding (int or tuple): 
            number of zeros to apply to the input image vertically and horizontally (h, w). If integer then defaults to (a, a)

    ## Shapes
        Input:
            (N, channels, in_height, in_width)

        Output
            (N, channels, out_height, out_width)
    """
    def __init__(
            self, 
            kernel_size:int|tuple,
            stride:int|tuple,
            padding:int|tuple
            ):
        super().__init__()
        self._save_hparams(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        
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
    
    def parameters(self):
        return []
    
    def get_state(self):
        pass

    def load_state(self, state):
        pass