from ..core.tensor import Tensor
import numpy as np


def relu(x:Tensor):
    relu = np.maximum(0, x.data)
    out = Tensor(relu, requires_grad=True)

    if x.requires_grad:
        def _relu_backward():
            x.grad += out.grad * (x.data > 0)

        out._parents = {x,}
        out._backward_fn = _relu_backward

    return out