import numpy as np
from . import layers
from .ops import data
from .core.tensor import Tensor
from .graph import Graph
import pickle
from matplotlib import pyplot as plt
from copy import deepcopy
import gc
import time

class Model:
    def __init__(self):
        self.layers = []
        self.layer_count = 0

    def add_layer(self, layer):
        self.layers.append(layer)
        self.layer_count += 1

    def compile(self, optimizer, loss):
        self.loss = loss

        for layer in self.layers:
            if hasattr(layer, "initialize_parameters"):
                layer.initialize_optimizers(optimizer)

    def feed_forward(self, input_data:Tensor, aux_input:Tensor=None):
        activations = []
        for i in range(self.layer_count):
            input = input_data if i == 0 else activations[i - 1]
            if isinstance(self.layers[i], layers.Concatenate):
                input = Tensor(np.concatenate((input.data, aux_input.data), axis=1))
            layer_activations = self.layers[i].forward(input)

            activations.append(layer_activations)
        return activations

    def backprop(self, loss:Tensor):
        loss.backward()

        for layer in self.layers:
            if hasattr(layer, "optimize_parameters"):
                layer.optimize_parameters()
            
    def predict(self, input:Tensor, aux_input=None):
        outputs = self.feed_forward(input, aux_input)
        return outputs[-1]
    
    def validation_loss(self, test_x, test_y, epoch):
        batch_count = len(test_x)
        loss = np.zeros(batch_count)
        for i in range(batch_count):
            output = self.predict(test_x[i])
            loss[i] = np.sum(self.loss.loss(output, test_y[i]))
            print(f"Epoch: {epoch} - Validation Batch: {i} / {batch_count}", end="\r")
        loss = np.mean(loss)
        return loss
    
    def fit(
            self, 
            x:Tensor=None, 
            y:Tensor=None, 
            aux=None,
            epochs:int=1, 
            batch_size:int=32, 
            shuffle:bool=True, 
            use_graph:bool=False, 
            graph_rate:int=1,
    ):
        if not isinstance(x, Tensor) or not isinstance(y, Tensor):
            raise TypeError("x and y labels must be tensors")
        
        if not isinstance(aux, Tensor):
            aux = Tensor(np.zeros(shape=x.data.shape))

        sample_count = x.shape[0]
        
        if use_graph:
            graph = Graph("Epochs", "Loss", "Loss Over Time")
            graph.add_plot("Loss")
        
        for epoch in range(epochs):
            train_x = Tensor(x.data)
            train_y = Tensor(y.data)
            epoch_loss = 0

            if shuffle:
                seed = np.random.randint(0, 10000)
                train_x.data = data.shuffle_data(train_x.data, seed)
                train_y.data = data.shuffle_data(train_y.data, seed)
                aux.data = data.shuffle_data(aux.data, seed)

            x_batches = data.create_batches(train_x, batch_size)
            y_batches = data.create_batches(train_y, batch_size)
            aux_batches = data.create_batches(aux, batch_size)
            batch_num = sample_count // batch_size

            epoch_loss = 0
            for batch in range(batch_num):
                start_time = time.perf_counter()
                batch_loss = self.fit_batch(x_batches[batch], y_batches[batch], aux_batches[batch])
                epoch_loss += batch_loss
                print(f"Epoch: {epoch+1} - Loss: {batch_loss:.5f} - Batch: {batch + 1} / {batch_num} - Batch Time: {time.perf_counter() - start_time:.3f}", end="\r")
                #print(len(gc.get_objects()))
            epoch_loss /= batch_num
            print(f"Epoch: {epoch + 1} - Loss: {epoch_loss:.5f}")
            
            
            if epoch % graph_rate == 0 and use_graph:
                graph.append("Loss", epoch, epoch_loss)
                graph.update()
    
    def fit_batch(self, x_batch, y_batch, aux_input=None):
        activations = self.feed_forward(x_batch, aux_input)
        loss = self.loss(activations[-1], y_batch)

        self.backprop(loss)
        gc.collect()
        return loss.data[0]
        
    def parameters(self):
        for layer in self.layers:
            yield layer.weights
            yield layer.biases
    
    def save(self, filepath:str="/model_struct"):
        model_data = {
            "layers": [layer for layer in self.layers],
            "loss": self.loss,
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
        self.loss = model_data["loss"]