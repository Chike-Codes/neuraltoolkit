from cProfile import label

import numpy as np
from neuraltoolkit.core.tensor import Tensor

class BinaryCrossEntropy:
    def __call__(self, predictions:Tensor, labels:Tensor):
        epsilon = 1e-5

        batch_size = predictions.shape[0]

        BCE = -(labels.data * np.log(predictions.data + epsilon) + (1 - labels.data) * np.log(1 - predictions.data + epsilon))
        BCE = np.sum(BCE) / batch_size
        out = Tensor(BCE, requires_grad=True)
        
        def _BCE_backward():
            predictions.grad += (predictions.data - labels.data + epsilon) * out.grad * (1 / batch_size)

        out._parents = {predictions}
        out._backward_fn = _BCE_backward

        return out