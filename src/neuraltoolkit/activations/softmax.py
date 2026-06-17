from ..core.tensor import Tensor
import numpy as np


def softmax(x:Tensor):
    expos = np.exp(x.data - np.max(x.data, axis=1, keepdims=True)) # normalized
    expo_sum = np.sum(expos, axis=1, keepdims=True)
    softmax = expos / expo_sum

    out = Tensor(softmax, requires_grad=True)

    if x.requires_grad:
        def _softmax_backward():
            dot = (out.grad * softmax).sum(axis=1, keepdims=True)
            x.grad += softmax * (out.grad - dot)
        
        out._parents = {x,}
        out._backward_fn = _softmax_backward

    return out