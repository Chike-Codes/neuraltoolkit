from abc import ABC, abstractmethod

class Optimizer(ABC):
    @abstractmethod
    def optimize(self):
        pass

    def clear_grad(self):
        for param in self.parameters:
            param.clear_grad()