from abc import ABC, abstractmethod

class Optimizer(ABC):
    """`Optimizer` is the base class that all of NTK's optimizers draw from. 
    An otpimizer is the algorithm that updates a model's weights and biases.
    It determins how to adjust the parameters based on their gradient and the 
    optimizers hyper-parameters"""

    @abstractmethod
    def optimize(self):
        """Applies the optimizer function to the model parameters"""
        pass

    def clear_grad(self):
        """Resets / sets the model gradients to zero so they can be updated again in the next training step."""
        for param in self.parameters:
            param.clear_grad()