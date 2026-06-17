import numpy as np
from neuraltoolkit.optimizers.optimizer import Optimizer


class RMSProp(Optimizer):
    """
    RMSProp optimizer.

    Adapts learning rates using a moving average of squared gradients.

    Args:
        parameters: Trainable model parameters.
        learning_rate: Step size used for parameter updates.
        decay_rate: Decay rate for the moving average.
    """
    def __init__(self, parameters, learning_rate = 0.001, decay_rate = 0.9):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = 1e-8
        
        self.parameters = parameters

        self.moving_averages = []
        for param in self.parameters:
            self.moving_averages.append(np.zeros(param.grad.shape))

    # updates weights and biases using Adagrad an adaptive optimizer that updates the LR for each parameter based on the sqrt of summed squares of previous gradients
    def optimize(self):
        for param, moving_average in zip(self.parameters, self.moving_averages):
            moving_average = self.decay_rate * moving_average + (1 - self.decay_rate) * param.grad ** 2
            param.data -= (self.learning_rate / (np.sqrt(moving_average) + self.epsilon)) * param.grad

    def clear_grad(self):
        for param in self.parameters:
            param.clear_grad()