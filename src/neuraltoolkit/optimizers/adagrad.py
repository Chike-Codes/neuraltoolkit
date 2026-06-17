import numpy as np
from neuraltoolkit.optimizers.optimizer import Optimizer


class Adagrad(Optimizer):
    """
    Adagrad optimizer.

    Adapts learning rates based on the history of observed gradients.

    Args:
        parameters: Trainable model parameters.
        learning_rate: Initial step size used for parameter updates.
    """
    def __init__(self, parameters, learning_rate = 0.001):
        self.learning_rate = learning_rate
        self.epsilon = 1e-8
        
        self.parameters = parameters

        self.learning_gradients = []
        for param in self.parameters:
             self.learning_gradients.append(np.zeros(param.grad.shape))

    # updates weights and biases using Adagrad an adaptive optimizer that updates the LR for each parameter based on the sqrt of summed squares of previous gradients
    def optimize(self):
        for param, learning_gradient in zip(self.parameters, self.learning_gradients):
            learning_gradient += param.grad ** 2
            param.data -= (self.learning_rate / (np.sqrt(learning_gradient) + self.epsilon)) * param.grad

    def clear_grad(self):
        for param in self.parameters:
            param.clear_grad()