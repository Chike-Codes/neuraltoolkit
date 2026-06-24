from neuraltoolkit.core.tensor import Tensor
from neuraltoolkit.modules.module import Module
from neuraltoolkit.optimizers.optimizer import Optimizer
from neuraltoolkit.training.history import History
from neuraltoolkit.training.trainer import Trainer
from neuraltoolkit.modules import Registry


class Sequential(Module):
    """Standard sequential neural network architure. 
    Includes fit() for simple training"""
    def __init__(self, *args):
        super().__init__()
        self.layers = []
        self.layer_count = 0

        for layer in args:
            if not isinstance(layer, Module):
                raise ValueError("Sequential must be initialized with Modules only.")
            self.add_layer(layer)
            
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
            layer_activations = self.layers[i](input)

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
    
    def get_config(self):
        self._save_hparams(
            layers=[layer.get_config() for layer in self.layers]
        )
        return self._config
    
    @classmethod
    def from_config(cls, config):
        layers = config["values"]["layers"]

        args = []
        for layer_config in layers:
            module = Registry.get(layer_config["name"]).from_config(layer_config)
            args.append(module)

        return cls(*args)
    
    def get_state(self):
        save_state = {
            "layers": [layer.get_state() for layer in self.layers]
        }

        return save_state
    
    def load_state(self, state):
        saved_layers = state["layers"]

        for layer, params in zip(self.layers, saved_layers):
            if params == None:
                continue

            layer.load_state(params)