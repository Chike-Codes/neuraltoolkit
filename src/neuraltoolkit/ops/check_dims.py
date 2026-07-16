from neuraltoolkit.core.tensor import Tensor


def check_dims(x:Tensor, dims:int):
    """Returns an error if the tensor doesn't match the expected dimensions"""
    if x.data.ndim != dims:
        raise ValueError(f"Expected tensor dimensions of {dims}, but recieved {x.data.ndim}")