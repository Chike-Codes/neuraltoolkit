from neuraltoolkit.modules.module import Module


class Activation(Module):
    def parameters(self):
        return []
    
    def get_state(self):
        pass

    def load_state(self, state):
        pass

    @classmethod
    def from_config(cls, config):
        return cls()