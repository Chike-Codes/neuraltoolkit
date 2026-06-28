from abc import ABC, abstractmethod
import warnings
from neuraltoolkit.core.parameter import Parameter
from neuraltoolkit.core.tensor import Tensor
from neuraltoolkit.modules import Registry
from importlib.metadata import version
import numpy as np
import json

class Module(ABC):
    """Parent class for all trainable objects"""

    def __init__(self):
        self._config = {
            "name":self.__class__.__name__
        }

    def __init_subclass__(cls):
        super().__init_subclass__()
        Registry.register(cls)

    def __call__(self, x):
        """Calls the forward method"""
        return self.forward(x)

    def _save_hparams(self, **kwargs):
        """Returns a dictionary with hyper-parameter configuration"""
        self._config["values"] = kwargs

    def get_config(self):
        """
        Returns a dictionary of hyper parameter configuration

        **Usage:**
            config = module.get_config()
            new_module = module.from_config(config)
        """
        return self._config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config["values"])
    
    def parameters(self):
        try:
            return (self.weights, self.biases)
        except:
            raise NotImplementedError(("The default definition for parameters()"
                                      "override this function for specific implementation"))
    
    def get_state(self):
        try:
            state = {
                "weights":self.weights.data.tolist(),
                "biases":self.biases.data.tolist()
            }
            return state
        except:
            raise NotImplementedError(("The default definition for parameters()"
                                      "override this function for specific implementation"))
    
    def load_state(self, state):
        try:
            self.weights = Parameter(np.array(state["weights"]))
            self.biases = Parameter(np.array(state["biases"]))
        except:
            raise NotImplementedError(("The default definition for parameters()"
                                      "override this function for specific implementation"))
    
    def save(self, path:str):
        save_dict = {
            "ntk_version":version("ntk-ml"),
            "module_type":self.__class__.__name__,
            "config":self.get_config(),
            "state":self.get_state()
        }

        with open(path, "w") as f:
            json.dump(save_dict, f, indent=4)

    def get_save_dict(self):
        save_dict = {
            "ntk_version":version("ntk-ml"),
            "module_type":self.__class__.__name__,
            "config":self.get_config(),
            "state":self.get_state()
        }

        return save_dict


    @classmethod
    def load(cls, path:str):
        with open(path, "r") as f:
            data = json.load(f)

        if data["module_type"] != cls.__name__:
            raise TypeError(("Attempting to load module data into the wrong module" 
                            f"Cannot load {data["module_type"]} into {cls.__name__}"))
        
        if data["ntk_version"] != version("ntk-ml"):
            warnings.warn((f"Model was saved with NTK {data["ntk_Version"]}"
                          f"but the current version is {version("ntk-ml")}"))

        module = cls.from_config(data["config"])
        module.load_state(data["state"])

        return module

    @abstractmethod
    def forward(self, x) -> Tensor:
        pass