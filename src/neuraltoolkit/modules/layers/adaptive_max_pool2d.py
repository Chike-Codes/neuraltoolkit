from neuraltoolkit.modules.module import Module
from ...core import tensor_ops
from ...ops.image_processing import calc_kernel_and_stride, get_im2col_indices

class Adaptive_Max_Pool2d(Module):
    """
    Scales down an image by selecting the maximum pixel given a desired output shape

    stride = input_size // output_size

    kernel_size = input_size - (output - 1) * stride

    Args:
        H_out (int): height of the output image
        W_out (int): width of the output image
    """
    def __init__(
            self,
            H_out:int,
            W_out:int
            ):
        super().__init__()
        self._save_hparams(
            H_out=H_out,
            W_out=W_out
        )
        
        self.H_out = H_out
        self.W_out = W_out

        self.flat_index_map = None
        self.prev_padded_shape = None

    def forward(self, x):
        """
        Performs a forward pass through the Adaptive Max Pool layer
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Tensor of shape (batch_size, out_channels, out_height, out_width)
        """
        N, C, H, W = x.shape
        kernel_size, stride = calc_kernel_and_stride(H, W, self.H_out, self.W_out)

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
    