from ..core.tensor import Tensor
import numpy as np


def leaky_relu(x:Tensor, alpha=0.1):
    relu = np.maximum(x.data * alpha, x.data)
    out = Tensor(relu, requires_grad=True)

    if x.requires_grad:
        def _relu_backward():
            x.grad += out.grad.copy()
            x.grad[x.data <= 0] *= alpha

        out._parents = {x,}
        out._backward_fn = _relu_backward

    return out