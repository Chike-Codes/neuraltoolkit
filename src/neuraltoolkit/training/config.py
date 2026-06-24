from dataclasses import dataclass
from neuraltoolkit.data.dataloader import Dataloader

@dataclass
class TrainingConfig:
    train_loader:Dataloader=None
    val_loader:Dataloader=None
    epochs:int=None
    verbose:bool=False