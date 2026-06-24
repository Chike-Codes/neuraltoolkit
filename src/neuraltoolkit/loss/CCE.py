import numpy as np
from ..core.tensor import Tensor

class CategoricalCrossEntropy:
    """
    Categorical cross entropy loss for multi-class classification.

    Expects raw logits as input and one-hot encoded labels.
    """
    def __call__(self, logits:Tensor, labels:Tensor):
        shifted_logits = logits.data - logits.data.max(axis=1, keepdims=True)

        exp = np.exp(shifted_logits)
        sum_exp = exp.sum(axis=1, keepdims=True)
        
        log_sum_exp = np.log(sum_exp)
        log_softmax = shifted_logits - log_sum_exp

        N = logits.data.shape[0]
        loss_val = -np.sum(labels.data * log_softmax) / N

        out = Tensor(loss_val, requires_grad=True)

        if logits.requires_grad:
            def _CE_backward():
                softmax = exp / sum_exp
                logits.grad += out.grad * (softmax - labels.data) / N

            out._parents = {logits}
            out._backward_fn = _CE_backward

        return out

 
def _cross_entropy_from_softmax(predictions:Tensor, labels:Tensor):
    log_probs = np.log(predictions.data + 1e-8)
    per_sample_loss = -np.sum(labels.data * log_probs, axis=1)
    average_loss = np.mean(per_sample_loss)

    out = Tensor(average_loss, requires_grad=True)

    if predictions.requires_grad:
        def _CCE_backward():
            eps = 1e-12
            predictions_safe = predictions.data + eps
            predictions.grad += out.grad * (-labels.data / predictions_safe)

        out._parents = {predictions, labels,}
        out._backward_fn = _CCE_backward

    return out