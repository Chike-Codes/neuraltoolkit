from ..graph import Graph
from dataclasses import dataclass


@dataclass
class Metrics:
    loss:float=None
    val_loss:float=None
    accuracy:float=None

class History:
    def __init__(self):
        self._epochs = 0
        self.loss = []
        self.val_loss = []
        self.accuracy = []

        self.tracker_map = {
            "loss":self.loss,
            "val_loss":self.val_loss,
            "accuracy":self.accuracy
        }

        self.colors = ["red", "green", "orange"]

    def plot(self, *args):
        plot = Graph("Metrics", "Time", "Metrics Over Time")
        
        for string, color in zip(args, self.colors):
            name = string.lower()
            plot.add_plot(name, list(range(self._epochs)), self.tracker_map[name], color)
        
        plot.show()

    def log(self, metrics:Metrics):
        self.loss.append(metrics.loss)
        self.val_loss.append(metrics.val_loss)
        self.accuracy.append(metrics.accuracy)

    @property
    def epochs(self):
        return self._epochs
    
    def epoch_step(self):
        self._epochs += 1
