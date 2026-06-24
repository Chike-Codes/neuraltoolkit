from neuraltoolkit.core.tensor import Tensor
from neuraltoolkit.modules.activations.activation import Activation
import numpy as np

class Sigmoid(Activation):
    def forward(self, x:Tensor):
        sig = 1 / (1 + np.exp(-x.data))
        out = Tensor(sig, requires_grad=True)

        if x.requires_grad:
            def _sig_backward():
                x.grad += out.grad * sig * (1 - sig)

            out._parents = {x,}
            out._backward_fn = _sig_backward

        return out