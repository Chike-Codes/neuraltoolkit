from .tensor import Tensor
from ..ops.image_processing import *
import numpy as np


def conv2d(x, kernel, stride, pad, flat_index_map) -> Tensor:
    """
    Performs a convolutional operation on a batch of images.
    --------------------------------------------------------

    Arguments:
        x (tensor (N, C_in, H, W)): Batch of input images.
        kernel (parameter (C_out, C_in, K, K)): kernel/filter weights.
        stride (int or tuple): The number pixels the kernel/filter moves at each step. Defaults to 1.
        pad (int or tuple): The number of filler pixels (0s) to be inserted around the borders of the images.
        flat_index_map: output from get_im2col_indices(). must match complete conv output shape
    ------------------------------------------------------------------------------------------------

    Returns:
        Tensor(N, C_out, out_height, out_width)
    """

    stride_h, stride_w = split_2d_param(stride)
    pad_h, pad_w = split_2d_param(pad)

    C_out = kernel.shape[0]
    K_h = kernel.shape[2] # kernel/filter height
    K_w = kernel.shape[3] # kernel/filter width

    N, C_in, H, W = x.shape

    if pad_h > 0 or pad_w > 0:
        x_padded = np.pad(x.data, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    else:
        x_padded = x.data
    out_height, out_width = output_image_size(H, W, (K_h, K_w), stride, pad)

    kernel_flat = np.reshape(kernel.data, shape=(C_out, C_in * K_h * K_w))

    img_patches = im2col_fast(x_padded, flat_index_map)

    out_flat = img_patches @ kernel_flat.T # shape (N, H_out * W_out, C_out)
    
    out = out_flat.reshape((N, out_height, out_width, C_out))
    out = out.transpose(0, 3, 1, 2).copy()
    out = Tensor(out, requires_grad=True)

    def _conv2d_backward():
        if x.requires_grad and Tensor.grad_enabled:
            grad_x = out.grad.copy() # (N, C_out, H_out, W_out)
            flat_grad_x = grad_x.reshape(N, C_out, out_height * out_width)
            flat_grad_x = flat_grad_x.transpose(0, 2, 1).copy()
            dx_col = flat_grad_x @ kernel_flat

            dx = col2im_fast(dx_col, flat_index_map, x_padded.shape)

            if pad_h > 0 or pad_w > 0:
                x.grad += unpad(dx, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)))
            else:
                x.grad += dx
        
        if kernel.requires_grad and Tensor.grad_enabled:
            grad = out.grad.copy() # (N, C_out, H_out, W_out)
            grad_reshaped = grad.reshape((N, C_out, out_height * out_width))

            # (N, C_out, H_out*W_out) @ (N H_out*W_out, C_in*Kh*kw) = (N, C_out, C_int*Kh*Kw)
            dw_batches = grad_reshaped @ img_patches

            # summed acrossed batches and set to original kernel shape
            dw = np.sum(dw_batches, axis=0).reshape(C_out, C_in, K_h, K_w) 
            
            kernel.grad += dw

    out._parents = {x, kernel}
    out._backward_fn = _conv2d_backward
    return out

def reshape(x, shape):
    x_shape = x.shape
    data = x.data.copy()
    out = np.reshape(data, shape=shape)
    out = Tensor(out, requires_grad=True)

    out._parents = {x}

    def _reshape_backward():
        if x.requires_grad and Tensor.grad_enabled:
            grad_x = out.grad.copy()
            x.grad += np.reshape(grad_x, shape=x_shape)
    
    out._backward_fn = _reshape_backward
    return out

def max_pool2d(
        x:Tensor,
        kernel_size:int|tuple,
        stride:int|tuple,
        pad:int|tuple,
        flat_index_map
):
    """
    Performs a max pooling operation on a batch of images.
    -----------------------------------------------------

    Arguments:
        x (tensor of shape: (N, C_in, H, W)): A batch of input images.
        kernel_size (int or tuple): The size of kernel/filter.
        stride (int or tuple): The number pixels the kernel/filter moves at each step. Defaults to 1.
        pad (int or tuple): The number of filler pixels (0s) to be inserted around the borders of the images.
        flat_index_map: output from get_im2col_indices(). must match complete max_pool output shape      
    Returns:
        scaled down numpy array of max values
    """


    kh, kw = split_2d_param(kernel_size)
    kernel_area = kh * kw

    N, C, H, W = x.shape # N: samples, C: channels, H: height, W: width
    pad_h, pad_w = split_2d_param(pad)

    if pad_h > 0 or pad_w > 0:
        padded_x = np.pad(x.data, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    else:
        padded_x = x.data
    H_out, W_out = output_image_size(H, W, kernel_size, stride, pad)

    HW_out = H_out*W_out


    patches_flat = im2col_fast(padded_x, flat_index_map) # Shape (N, H*W, C*K*K)
    patches_flat = np.reshape(patches_flat, shape=(N, HW_out, C, kernel_area))
    #patches_flat = patches_flat.transpose(0, 2, 1, 3) # Shape (N, C, HW_out, K*K)

    max_val_indices = np.argmax(patches_flat, axis=-1, keepdims=True) # Shape (N, HW_out, C, 1)
    max_pool = np.take_along_axis(patches_flat, max_val_indices, axis=-1) # Shape (N, HW_out, C, 1)
    max_pool = np.squeeze(max_pool, axis=-1) # Shape (N, HW_out C,)
    max_pool = max_pool.transpose(0, 2, 1).copy()
    max_pool = max_pool.reshape(N, C, H_out, W_out)

    out = Tensor(max_pool, requires_grad=True)

    out._parents = {x}

    def _max_pool2d_backward():
        if x.requires_grad and Tensor.grad_enabled:
            max_indices = max_val_indices.squeeze(-1) # (N, HW_out, C)

            n_idx = np.arange(N)[:, None, None]
            hw_idx = np.arange(HW_out)[None, :, None]
            c_idx = np.arange(C)[None, None, :]

            flat_index_map_reshaped = flat_index_map.reshape(N, HW_out, C, kernel_area)

            # Shape (N, HW_out, C) - Advanced indexing
            im_index_map = flat_index_map_reshaped[n_idx, hw_idx, c_idx, max_indices]

            # Shape (N, C, HW_out)
            im_index_map = im_index_map.transpose(0, 2, 1).copy()

            grad_reshaped = out.grad.reshape((N, C, HW_out))
            
            dx_flat = np.bincount(im_index_map.ravel(), grad_reshaped.ravel(), minlength=np.prod(padded_x.shape))
            dx = dx_flat.reshape(N, C, H, W)

            if pad_h > 0 or pad_w > 0:
                x.grad += unpad(dx, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)))
            else:
                x.grad += dx


    out._backward_fn = _max_pool2d_backward
    return out