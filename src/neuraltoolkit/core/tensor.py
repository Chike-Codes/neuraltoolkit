import numpy as np
from .dtype import Dtype
from .device import Device

class Tensor:
    """
    Standard multidemsional datastorage

    Args:
        data (numpy array or list): Tensor data (lists are converted to numpy arrays)
        requires_grad (bool): Whether the tensor tracks gradients (defaults to False)
        
    """
    data: np.ndarray
    shape: tuple[int, ...]
    dtype: Dtype

    grad_enabled=True

    def __init__(self, data, requires_grad=False):
        self._parents = set()
        self._backward_fn = None

        self.data = self._init_data(data)

        self.shape = self.data.shape

        if self.data.dtype != np.float32:
            self.data = self.data.astype(np.float32)
        self.dtype = self.data.dtype
        
        self.requires_grad = requires_grad

        self.grad = np.zeros(shape=self.shape, dtype=np.float32) if requires_grad else None

        self.name = "Tensor"

    def _init_data(self, d):
        data = None
        if isinstance(d, np.ndarray):
            data = d
        elif isinstance(d, list):
            data = np.array(d)
        else:
            data = np.array([d])

        if not np.issubdtype(data.dtype, np.number):
            raise TypeError("Tensor data must be numeric")
        
        return data
    
    def clear_grad(self):
        if self.requires_grad and Tensor.grad_enabled:
            self.grad *= 0

    def backward(self):
        topo = []
        visited = set()

        self.grad = np.ones_like(self.data)

        def build(node):
            if node not in visited:
                visited.add(node)

                for parent in node._parents:
                    build(parent)
                topo.append(node)

        build(self)


        for node in reversed(topo):
            if node._backward_fn:
                node._backward_fn()
                self._clear_links()

    def _clear_links(self):
        self._parents = set()
        self._backward_fn = None

    @property
    def T(self):
        out = Tensor(self.data.T, requires_grad=self.requires_grad)

        if self.requires_grad and Tensor.grad_enabled:
            def _transpose_backward():
                self.grad += out.grad.T

            out._parents = {self}
            out._backward_fn = _transpose_backward

        return out
    
    @staticmethod
    def _reduce_broadcast(grad, shape):
        # Remove extra leading dims
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)

        # Collapse broadcasted axes
        for axis, size in enumerate(shape):
            if size == 1:
                grad = grad.sum(axis=axis, keepdims=True)

        return grad

    def __repr__(self):
        return f"Tensor:\n {self.data} \n"
    
    def _Tensor_wrapper(self, other):
        return other if isinstance(other, Tensor) else Tensor(other)
    
    def __getitem__(self, idx):
        sliced_data = self.data[idx]

        out = Tensor(sliced_data, requires_grad=self.requires_grad)

        def _slice_backward():
            if self.requires_grad and Tensor.grad_enabled:
                self.grad[idx] += out.grad

            out._parents = {self}
            out._backward_fn = _slice_backward

        return out
    
    def __add__(self, other):
        other = self._Tensor_wrapper(other)
        out = Tensor(self.data + other.data, requires_grad=True)

        out._parents = {self, other}

        def _add_backward():
            if self.requires_grad and Tensor.grad_enabled:
                grad_self = out.grad.copy()
                self.grad += self._reduce_broadcast(grad_self, self.shape)
            
            if other.requires_grad:
                grad_other = out.grad.copy()
                other.grad += self._reduce_broadcast(grad_other, other.shape)
        
        out._backward_fn = _add_backward
        return out
    
    def __radd__(self, other):
        other = self._Tensor_wrapper(other)
        out = Tensor(other.data + self.data, requires_grad=True)

        out._parents = {self, other}

        def _add_backward():
            if self.requires_grad and Tensor.grad_enabled:
                grad_self = out.grad.copy()
                self.grad += self._reduce_broadcast(grad_self, self.shape)
            
            if other.requires_grad:
                grad_other = out.grad.copy()
                other.grad += self._reduce_broadcast(grad_other, other.shape)

        out._backward_fn = _add_backward
        return out
    
    def __sub__(self, other):
        other = self._Tensor_wrapper(other)
        out = Tensor(self.data - other.data, requires_grad=True)

        out._parents = {self, other}

        def _sub_backward():
            if self.requires_grad and Tensor.grad_enabled:
                grad_self = out.grad.copy()
                self.grad += self._reduce_broadcast(grad_self, self.shape)
            
            if other.requires_grad:
                grad_other = out.grad.copy()
                other.grad -= self._reduce_broadcast(grad_other, other.shape)

        out._backward_fn = _sub_backward
        return out
    
    def __rsub__(self, other):
        other = self._Tensor_wrapper(other)
        out = Tensor(other.data - self.data, requires_grad=True)

        out._parents = {self, other}

        def _sub_backward():
            if self.requires_grad and Tensor.grad_enabled:
                grad_self = out.grad.copy()
                self.grad -= self._reduce_broadcast(grad_self, self.shape)
            
            if other.requires_grad:
                grad_other = out.grad.copy()
                other.grad += self._reduce_broadcast(grad_other, other.shape)

        out._backward_fn = _sub_backward
        return out
    
    def __mul__(self, other):
        other = self._Tensor_wrapper(other)
        out = Tensor(self.data * other.data, requires_grad=True)

        out._parents = {self, other}

        def _mul_backward():
            if self.requires_grad and Tensor.grad_enabled:
                grad_self = out.grad * other.data
                self.grad += self._reduce_broadcast(grad_self, self.shape)

            if other.requires_grad:
                grad_other = out.grad * self.data
                other.grad += self._reduce_broadcast(grad_other, other.shape)

        out._backward_fn = _mul_backward
        return out
    
    def __rmul__(self, other):
        other = self._Tensor_wrapper(other)
        out = Tensor(other.data * self.data, requires_grad=True)

        out._parents = {self, other}

        def _mul_backward():
            if self.requires_grad and Tensor.grad_enabled:
                grad_self = out.grad * other.data
                self.grad += self._reduce_broadcast(grad_self, self.shape)

            if other.requires_grad:
                grad_other = out.grad * self.data
                other.grad += self._reduce_broadcast(grad_other, other.shape)

        out._backward_fn = _mul_backward
        return out
    
    def __truediv__(self, other):
        other = self._Tensor_wrapper(other)
        out = Tensor(self.data / other.data, requires_grad=True)

        out._parents = {self, other}

        def _div_backward():
            if self.requires_grad and Tensor.grad_enabled:
                grad_self = out.grad / other.data
                self.grad += self._reduce_broadcast(grad_self, self.shape)

            if other.requires_grad:
                grad_other = -out.grad * self.data / (other.data ** 2)
                other.grad += self._reduce_broadcast(grad_other, other.shape)

        out._backward_fn = _div_backward
        return out
    
    def __rtruediv__(self, other):
        other = self._Tensor_wrapper(other)
        out = Tensor(other.data / self.data, requires_grad=True)

        out._parents = {self, other}

        def _div_backward():
            if self.requires_grad and Tensor.grad_enabled:
                grad_self = -out.grad * other.data / (self.data ** 2)
                self.grad += self._reduce_broadcast(grad_self, self.shape)

            if other.requires_grad:
                grad_other = out.grad / self.data
                other.grad += self._reduce_broadcast(grad_other, other.shape)

        out._backward_fn = _div_backward
        return out
    
    def __pow__(self, other):
        other = self._Tensor_wrapper(other)
        out = Tensor(self.data ** other.data, requires_grad=True)

        out._parents = {self, other}

        def _pow_backward():
            if self.requires_grad and Tensor.grad_enabled:
                grad_self = other.data * (self.data ** (other.data - 1)) * out.grad
                self.grad += self._reduce_broadcast(grad_self, self.shape)

            if other.requires_grad:
                grad_other = np.log(self.data) * (self.data ** other.data) * out.grad
                other.grad += self._reduce_broadcast(grad_other, other.shape)

        out._backward_fn = _pow_backward
        return out
    
    def __rpow__(self, other):
        other = self._Tensor_wrapper(other)
        out = Tensor(other.data ** self.data, requires_grad=True)

        out._parents = {self, other}

        def _pow_backward():
            if self.requires_grad and Tensor.grad_enabled:
                grad_self = np.log(other.data) * (other.data ** self.data) * out.grad
                self.grad += self._reduce_broadcast(grad_self, self.shape)

            if other.requires_grad:
                grad_other = self.data * (other.data ** (self.data - 1)) * out.grad
                other.grad += self._reduce_broadcast(grad_other, other.shape)

        out._backward_fn = _pow_backward
        return out
    
    def __matmul__(self, other):
        other = self._Tensor_wrapper(other)
        out = Tensor(self.data @ other.data, requires_grad=True)

        out._parents = {self, other}

        def _matmul_backward():
            if self.requires_grad and Tensor.grad_enabled:
                self.grad += out.grad @ other.data.T

            if other.requires_grad:
                other.grad += self.data.T @ out.grad

        out._backward_fn = _matmul_backward
        return out
    
    def __rmatmul__(self, other):
        other = self._Tensor_wrapper(other)
        out = Tensor(other.data @ self.data, requires_grad=True)

        out._parents = {self, other}

        def _matmul_backward():
            if self.requires_grad and Tensor.grad_enabled:
                self.grad += other.data.T @ out.grad

            if other.requires_grad:
                other.grad += out.grad @ self.data.T

        out._backward_fn = _matmul_backward
        return out

    def __floordiv__(self, other):
        other = self._Tensor_wrapper(other)
        out = Tensor(self.data // other.data)
    
    def __rfloordiv__(self, other):
        other = self._Tensor_wrapper(other)
        out = Tensor(other.data // self.data)
    
    def __mod__(self, other):
        other = self._Tensor_wrapper(other)
        out = Tensor(self.data % other.data)
    
    def __rmod__(self, other):
        other = self._Tensor_wrapper(other)
        out = Tensor(other.data % self._unwrap)