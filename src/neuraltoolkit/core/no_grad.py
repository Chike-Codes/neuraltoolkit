from .tensor import Tensor

class no_grad:
    def __enter__(self):
        self.previous = Tensor.grad_enabled
        Tensor.grad_enabled = False

    def __exit__(self, exc_type, exc, tb):
        Tensor.grad_enabled = self.previous