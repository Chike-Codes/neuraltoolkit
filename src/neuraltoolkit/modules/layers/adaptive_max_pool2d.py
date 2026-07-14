from neuraltoolkit.modules.module import Module
from ...core import tensor_ops
from ...ops.image_processing import calc_kernel_and_stride, get_im2col_indices

class Adaptive_Max_Pool2d(Module):
    """
    Scales down an image by selecting the maximum pixel given a desired output shape

    Args:
        height (int): 
            height of the output image

        width (int): 
            width of the output image

    ## Shapes
        Input:
            (N, channels, in_height, in_width)

        Output
            (N, channels, out_height, out_width)
    """
    def __init__(
            self,
            height:int,
            width:int
            ):
        super().__init__()
        self._save_hparams(
            height=height,
            width=width
        )
        
        self.height = height
        self.width = width

        self.flat_index_map = None
        self.prev_padded_shape = None

    def forward(self, x):
        N, C, H, W = x.shape
        kernel_size, stride = calc_kernel_and_stride(H, W, self.height, self.width)

        if x.shape != self.prev_padded_shape:
            self.flat_index_map = get_im2col_indices(x.shape, kernel_size, stride)
            self.prev_padded_shape = x.shape

        z = tensor_ops.max_pool2d(x, kernel_size, stride, 0, self.flat_index_map)
        return z
    
    def parameters(self):
        return []
    
    def get_state(self):
        pass

    def load_state(self, state):
        pass
    