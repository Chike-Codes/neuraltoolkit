import numpy as np
from ..core.tensor import Tensor


def mean_squared_error(predictions:Tensor, labels:Tensor):
    element_num = predictions.data.size
    error = predictions.data - labels.data
    MSE = np.mean(np.power(error, 2))
    out = Tensor(MSE, requires_grad=True)

    if predictions.requires_grad:
        def _MSE_backward():
            predictions.grad += out.grad * error * (2 / element_num)

        out._parents = {predictions}
        out._backward_fn = _MSE_backward

    return out