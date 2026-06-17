import numpy as np
from neuraltoolkit.optimizers.optimizer import Optimizer


class Adam(Optimizer):
    """
    Adaptive Moment Estimation (Adam) optimizer.

    Combines momentum and adaptive learning rates using moving
    averages of gradients and squared gradients.

    Args:
        parameters: Trainable model parameters.
        learning_rate: Step size used for parameter updates.
        beta1: Decay rate for the first moment estimate.
        beta2: Decay rate for the second moment estimate.
    """
    def __init__(self, parameters, learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999):
        # moving average decay for weights and biases for both first and second moments
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = 1e-8
        self.time = 0

        self.parameters = parameters

        self.momentums = []
        self.velocities = []
        for param in self.parameters:
            self.momentums.append(np.zeros(param.grad.shape))
            self.velocities.append(np.zeros(param.grad.shape))

    # updates weights and biases using Adagrad an adaptive optimizer that updates the LR for each parameter based on the sqrt of summed squares of previous gradients
    def optimize(self):
        for param, momentum, velocity in zip(self.parameters, self.momentums, self.velocities):
            self.time += 1
            momentum = self.beta1 * momentum + (1 - self.beta1) * param.grad

            velocity = self.beta2 * velocity + (1 - self.beta2) * (param.grad ** 2)

            #bias corrected estimates
            EM = momentum / (1 - self.beta1 ** self.time)
            EV = velocity / (1 - self.beta2 ** self.time)

            param.data -= self.learning_rate * (EM / (np.sqrt(EV) + self.epsilon))

    def clear_grad(self):
        for param in self.parameters:
            param.clear_grad()