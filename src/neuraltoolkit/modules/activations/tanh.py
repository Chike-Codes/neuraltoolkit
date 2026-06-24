from neuraltoolkit.core.tensor import Tensor
from neuraltoolkit.modules.activations.activation import Activation
import numpy as np

class Tanh(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x:Tensor):
        tanh = np.tanh(x.data)
        out = Tensor(tanh, requires_grad=True)

        if x.requires_grad:
            def _tanh_backward():
                x.grad += out.grad * (1 - np.power(tanh, 2))

            out._parents = {x,}
            out._backward_fn = _tanh_backward

        return out