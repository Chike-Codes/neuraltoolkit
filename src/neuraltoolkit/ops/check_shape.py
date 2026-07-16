from neuraltoolkit.core.tensor import Tensor


def check_shape(x:Tensor, shape:tuple):
    """Returns an error if the tensor doesn't match the expected shape"""

    if x.shape != shape:
        raise ValueError(f"Expected tensor Shape of {shape}, but recieved {x.shape}")