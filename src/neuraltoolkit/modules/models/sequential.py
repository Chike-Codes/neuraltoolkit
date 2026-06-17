import numpy as np
import neuraltoolkit as ntk
from neuraltoolkit.core.tensor import Tensor
from neuraltoolkit.modules.module import Module
from neuraltoolkit.optimizers.optimizer import Optimizer
from neuraltoolkit.training.history import History
from neuraltoolkit.training.trainer import Trainer
import pickle
from matplotlib import pyplot as plt
from copy import deepcopy
from neuraltoolkit import CLI
import gc

class Sequential(Module):
    """Standard sequential neural network architure. 
    Includes fit() for simple training"""
    def __init__(self):
        self.layers = []
        self.layer_count = 0

        self.optimizer = None
        self.loss_func = None

    def __call__(self, x:Tensor):
        return self.forward(x)

    def add_layer(self, layer):
        self.layers.append(layer)
        self.layer_count += 1

    def parameters(self):
        params = []
        for layer in self.layers:
            for p in layer.parameters():
                params.append(p)
        return params
    
    def compile(self,
                optimizer:Optimizer,
                loss
        ):
        self.optimizer = optimizer
        self.loss_func = loss
        
    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_loss_func(self, loss_func):
        self.loss_func = loss_func

    def forward(self, input_data:Tensor):
        activations = []
        for i in range(self.layer_count):
            input = input_data if i == 0 else activations[i - 1]
            layer_activations = self.layers[i].forward(input)

            activations.append(layer_activations)
        return activations[-1]
    
    def fit(
            self,
            data,
            y=None,
            *,
            epochs=1,
            batch_size=None,
            shuffle:bool=None,
            drop_remainder_batch:bool=None,
            validation_split=None,
            validation_data=None

    ) -> History:
        if self.optimizer == None or self.loss_func == None:
            raise ValueError("optimizer and loss function must be initialized")
        trainer = Trainer(self, self.optimizer, self.loss_func)
        history = trainer.fit(
            data,
            y,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_remainder_batch=drop_remainder_batch,
            validation_split=validation_split,
            validation_data=validation_data
        )

        return history
    
    def save(self, filepath:str="/model_struct"):
        model_data = {
            "layers": [layer for layer in self.layers],
            "loss": self.loss_func,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
        print("Model Saved")
    
    def load(self, filepath:str):
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        for layer_data in model_data["layers"]:
            layer = deepcopy(layer_data)
            self.layers.append(layer)
            self.layer_count += 1
        self.loss_func = model_data["loss"]