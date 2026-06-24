from neuraltoolkit.modules import module


_registry = {}

def register(name):
    if not issubclass(name, module.Module):
        raise ValueError("Can only register ntk.Module classes")
    
    registry_key = name.__name__

    if registry_key in _registry:
        raise ValueError(f"Duplicate Registration for {registry_key}")
    
    _registry[registry_key] = name

    return name

def get(name:str):
    if not name in _registry:
        raise ValueError(f"{name} does not exist with the ntk registry")
    
    return _registry[name]