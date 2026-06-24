from neuraltoolkit.core.tensor import Tensor
from neuraltoolkit.modules.activations.activation import Activation
import numpy as np


class Leaky_relu(Activation):
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def forward(self, x:Tensor):
        relu = np.maximum(x.data * self.alpha, x.data)
        out = Tensor(relu, requires_grad=True)

        if x.requires_grad:
            def _relu_backward():
                x.grad += out.grad.copy()
                x.grad[x.data <= 0] *= self.alpha

            out._parents = {x,}
            out._backward_fn = _relu_backward

        return out