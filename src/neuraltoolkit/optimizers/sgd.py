import numpy as np
from neuraltoolkit.optimizers.optimizer import Optimizer


class Sgd(Optimizer):
    """
    Stochastic Gradient Descent optimizer.

    Updates parameters using gradients and optional momentum.

    Args:
        parameters: Trainable model parameters.
        learning_rate: Step size used for parameter updates.
        momentum_rate: Momentum factor used to smooth updates.
    """
    def __init__(self, parameters, learning_rate = 0.001, momentum_rate = 0.9):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.parameters = parameters

        self.velocities = []
        for param in self.parameters:
            self.velocities.append(np.zeros(param.grad.shape))

    # updates weights and biases using stochastic gradient descent with momentum
    def optimize(self):
        for param, velocity in zip(self.parameters, self.velocities):
            velocity = (self.momentum_rate * velocity + (1 - self.momentum_rate) * param.grad) * self.learning_rate
            param.data -= self.velocity
    
    def clear_grad(self):
        for param in self.parameters:
            param.clear_grad()