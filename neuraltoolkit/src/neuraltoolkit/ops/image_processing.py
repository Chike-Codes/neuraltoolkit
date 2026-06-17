import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# image processing ----------------------------------------

def im2col(x, kernel_size, stride=1, pad=0):
    """
        Takes a batch of images and returns the batch as matrices organized by kernel patches by column.
        ------------------------------------------------------------------------------------------------

        Arguments:
            x (numpy array of shape: (N, C_in, H, W)): A batch of input images.
            kernel_size (int or tuple): The size of kernel/filter.
            stride (int or tuple): The number pixels the kernel/filter moves at each step. Defaults to 1.
            pad (int or tuple): The number of filler pixels (0s) to be inserted around the borders of the images.
        ------------------------------------------------------------------------------------------------

        Returns:
            matrix (N, H_out*W_out, C_in*K*K)
    """

    kernel_height, kernel_width = split_2d_param(kernel_size)

    stride_h, stride_w = split_2d_param(stride)
    stride_h = None if stride_h == 0 else stride_h
    stride_w = None if stride_w == 0 else stride_w

    pad_h, pad_w = split_2d_param(pad)

    N = x.shape[0] # number of images in batch
    C_in = x.shape[1] # number of input channels

    input_height = x.shape[2] # image height
    input_width = x.shape[3] # image width

    out_height = (input_height + 2 * pad_h - kernel_height) // stride_h + 1 # output image height
    out_width = (input_width + 2 * pad_w - kernel_width) // stride_w + 1 # output image width

    # pad images
    padded_samples = np.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    
    #matrix of kernel/filter patches
    out = sliding_window_view(padded_samples, window_shape=(C_in, kernel_height, kernel_width), axis=(1, 2, 3))[:, :, ::stride_h, ::stride_w]

    # reshape matrix
    out = np.reshape(out, shape=(N, out_height * out_width, C_in * kernel_height * kernel_width))

    return out

def col2im(x, shape, kernel_size, stride, pad):
    """
        Inverses the output from im2col.
        ------------------------------------------------------------------------------------------------

        Arguments:
            x (numpy array of shape: (N, H_out*W_out, C_in*kernel_height*kernel_width)): The output from im2col().
            shape (tuple (N, C_in, H, W)): shape of image data to be output. Should match what was input into im2col().
            kernel_size (int or tuple): The size of kernel/filter.
            stride (int or tuple): The number pixels the kernel/filter moves at each step. Defaults to 1.
            pad (int or tuple): The number of filler pixels (0s) to be inserted around the borders of the images.
        ------------------------------------------------------------------------------------------------

        Returns:
            matrix (N, C_in, H, W)
    """

    if isinstance(kernel_size, int):
        kernel_height = kernel_size
        kernel_width = kernel_size
    elif isinstance(kernel_size, tuple):
        kernel_height = kernel_size[0]
        kernel_width = kernel_size[1]
    else:
        raise TypeError("kernel_size must be an int or tuple")
    
    if isinstance(stride, int):
        stride_h = stride
        stride_w = stride
    else:
        stride_h = stride[0]
        stride_w = stride[1]

    if isinstance(pad, int):
        pad_h = pad
        pad_w = pad
    else:
        pad_h = pad[0]
        pad_w = pad[1]

    N = shape[0]
    input_channels = shape[1]
    original_height = shape[2]
    original_width = shape[3]

    x_width = (original_width + 2 * pad_w - kernel_width) // stride_w + 1 # with of x images

    x_reshaped = np.reshape(x, shape=(x.shape[0], x.shape[1], input_channels, kernel_height, kernel_width)) # splits the data by channel

    # blank output with the shape of the original image with padding applied
    # this ensures that all output pixels trace back. The padding is undone later
    out = np.zeros(shape=(N, input_channels, original_height + 2*pad_h, original_width + 2*pad_w))


    for n in range(N):
        for i, patch in enumerate(x_reshaped[n]):
            for c in range(input_channels):
                row = int(i / x_width) * stride_h
                col = i % x_width * stride_w

                out[n, c, row : row + kernel_height, col : col + kernel_width] += patch[c]

    #unpad padded values
    out = unpad(out, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)))
    return out

def im2col_fast(x, flat_index_map):
    return x.ravel()[flat_index_map]

def col2im_fast(dx_col, flat_index_map, x_shape):
    N, C, H, W = x_shape
    dx = np.bincount(flat_index_map.ravel(), weights=dx_col.ravel(), minlength=np.prod(x_shape))
    return dx.reshape(x_shape)


def unpad(x, pad_widths):
    """
    The inverse of numpy's pad().
    Removes values up to a certain depth from each side of the axes provided.
    -------------------------------------------------------------------------
    
    Arguments:
        x (numpy array): The ndarray to be unpadded.
        pad_widths (tuple): A tuple of tuples for the pad widths of each dimension. i.e. ((a, b), (c, d)...).
    -----------------------------------------------------------------------------------------------------

    Returns:
        numpy array with the specified padded values removed.
    """

    slices =  []
    for start, end in pad_widths:

        # correct for invalid pad values
        start = start if start > 0 else 0
        end = -end if end > 0 else None

        slices.append(slice(start, end))

    return x[tuple(slices)]

def split_2d_param(param) -> tuple:
    """
    Returns height and width of paramater separately.
    -------------------------------------------------
    Arguments:
        param (int or tuple): the parameter to be slit.
            if it's an int then the value is copied. 
            if it's a tuple then it's split

    ---------------------------------------------------
    Returns:
        (H, W)
    """

    if isinstance(param, int):
        H = param
        W = param
    elif isinstance(param, tuple):
        H, W = param
    else:
        raise TypeError("Parameter must be an int or tuple.")
    
    return (H, W)

def output_image_size(H, W, kernel_size, stride, padding):
    """
    Returns the shape of an image after applying a kernel stride.
    -------------------------------------------------------------

    Arguments:
        H (int): The height of the input image.
        W (int): the width of the input image.
        kernel_size (int or tuple): size/shape of the kernel.
        stride (int or tuple): number of steps the kernel will take.
        padding (int or tuple): The number of filler pixels (0s) to be inserted around the borders of the images.
    -------------------------------------------------------------------------------------------------------------

    Returns:
        The height and width of the output image (H_out, W_out)
    """
    kernel_height, kernel_width = split_2d_param(kernel_size)
    pad_h, pad_w = split_2d_param(padding)
    stride_h, stride_w = split_2d_param(stride)

    out_height = (H + 2 * pad_h - kernel_height) // stride_h + 1 # output image height
    out_width = (W + 2 * pad_w - kernel_width) // stride_w + 1 # output image width

    return (out_height, out_width)

def get_im2col_indices(x_shape, kernel_size, stride):
    """
    Creates a flat vector that maps patch data back to the original image indices.
    ------------------------------------------------------------------------------
    Arguments:
        x_shape (tuple): the shape of the image data (N, C, H, W).
        kernel_size (int or tuple): size/shape of the kernel.
        stride (int or tuple): number of steps the kernel will take.
    Returns:
        a mapping over im2col output that points to the indices of the original images
        (N, H_out*W_out, C*kh*kw)

    """
    N, C, H, W = x_shape

    kh, kw = split_2d_param(kernel_size)
    sh, sw = split_2d_param(stride)

    kernel_offset = np.arange(kh).reshape(-1, 1) * W + np.arange(kw)
    kernel_offset = kernel_offset.flatten().reshape(1, -1)

    channel_offset = np.arange(C).reshape(-1, 1) * (H * W)
    window_offset = (channel_offset + kernel_offset).reshape(1, -1)

    y = np.arange(0, H - kh + 1, sh)
    x = np.arange(0, W - kw + 1, sw)
    slide_offset = (np.repeat(y, len(x)) * W + np.tile(x, len(y))).reshape(-1, 1)

    image_offset = slide_offset + window_offset

    batch_offset = np.arange(N).reshape(-1, 1, 1) * (C * H * W)

    flat_index_map = batch_offset + image_offset
    return flat_index_map

def calc_kernel_and_stride(
        H_in:int,
        W_in:int,
        H_out:int,
        W_out:int
        ):
    """
    calculates the stride and kernel values based on the desired output shape
    -----------------------------------------------------------
    Returns:
        scaled down numpy array of max values
    """

    if H_out > H_in or W_out > W_in:
        raise ValueError("H_out and W_out values can not be greater than input resolution")

    S_h = max(1, H_in // H_out)
    S_w = max(1, W_in // W_out)

    K_h = H_in - (H_out - 1) * S_h
    K_w = W_in - (W_out - 1) * S_w

    K_h = max(1, K_h)
    K_w = max(1, K_w)

    predicted_h = (H_in - K_h) // S_h + 1
    predicted_w = (W_in - K_w) // S_w + 1

    if predicted_h != H_out or predicted_w != W_out:
        raise ValueError("Adaptive Max Pool2d Failed!")

    return (K_h, K_w), (S_h, S_w)