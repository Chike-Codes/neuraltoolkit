from ..core.tensor import Tensor
import numpy as np


def tanh(x:Tensor):
    tanh = np.tanh(x.data)
    out = Tensor(tanh, requires_grad=True)

    if x.requires_grad:
        def _tanh_backward():
            x.grad += out.grad * (1 - np.power(tanh, 2))

        out._parents = {x,}
        out._backward_fn = _tanh_backward

    return out