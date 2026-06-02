from ..core.tensor import Tensor


def linear(x:Tensor):
    lin = x.data.copy()
    out = Tensor(lin, requires_grad=True)

    if x.requires_grad:
        def _lin_backward():
            x.grad += out.grad

        out._parents = {x, }
        out._backward_fn = _lin_backward

    return out