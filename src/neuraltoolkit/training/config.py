from dataclasses import dataclass

@dataclass
class TrainingConfig:
    train_loader=None
    val_loader=None
    epochs=None