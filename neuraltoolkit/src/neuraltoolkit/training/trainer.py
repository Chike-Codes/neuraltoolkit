import gc
import neuraltoolkit as ntk
from neuraltoolkit import CLI
from neuraltoolkit.core.tensor import Tensor
from neuraltoolkit.data.dataloader import Dataloader
from neuraltoolkit.data.dataset import Dataset
from neuraltoolkit.modules.module import Module
from neuraltoolkit.optimizers.optimizer import Optimizer
from neuraltoolkit.training.config import TrainingConfig
from neuraltoolkit.training.history import History, Metrics


class Trainer:
    """
    Manages model training and evaluation.

    Args:
        model: Model to train.
        optimizer: Optimizer used for parameter updates.
        loss: Loss function used during training.
    """
    def __init__(self,
                  model:Module,
                  optimizer:Optimizer,
                  loss
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss

    def _prepare_training_input(
            self,
            data,
            y,
            epochs,
            batch_size,
            shuffle,
            drop_remainder_batch,
            validation_split,
            validation_data
    ) -> TrainingConfig:
        config = TrainingConfig()

        val_loader = None

        # initializing dataloaders
        if isinstance(data, Dataloader):
            train_loader = data
        else:
            batch_size = 32 if batch_size == None else batch_size
            shuffle = False if shuffle == None else shuffle
            drop_remainder_batch = False if drop_remainder_batch == None else drop_remainder_batch

            if isinstance(data, Dataset):
                train_loader = Dataloader(data, batch_size, shuffle, drop_remainder_batch)
            elif isinstance(data, Tensor) and isinstance(y, Tensor):
                dataset = Dataset(data, y)
                train_loader = Dataloader(dataset, batch_size, shuffle, drop_remainder_batch)
            else:
                raise TypeError("The data argument must be a Dataloader, Dataset, or a Tensor pair with a Tensor for y")

        # handling conflicting arguments and Default Values
        if isinstance(data, Dataloader):
            if batch_size != None:
                raise ValueError(
                    "batch_size cannot be specified"
                    "when a Dataloader is provided"
                )
            if shuffle != None:
                raise ValueError(
                    "shuffle cannot be specified"
                    "when a Dataloader is provided"
                )
            if drop_remainder_batch != None:
                raise ValueError(
                    "drop_remainder_batch cannot be specified"
                    "when a Dataloader is provided"
                )
            
        # Validation data
        if validation_data != None and validation_split != None:
            raise ValueError("validation_data and validation_split cannot both be provided")
        if validation_split != None:
            if not isinstance(validation_split, float):
                raise ValueError("validation_split must be a float")
            
            train_loader, val_loader = train_loader.split(validation_split, shuffle)

        if isinstance(validation_data, Dataloader):
            val_loader = validation_data
        elif isinstance(validation_data, Dataset):
            # Does not need to be shuffled
            val_laoder = Dataloader(validation_data, batch_size, False, drop_remainder_batch)
        elif isinstance(validation_data, tuple) and all(isinstance(item, Tensor) for item in validation_data):
            val_x = validation_data[0]
            val_y = validation_data[1]

            val_dataset = Dataset(val_x, val_y)

            # Does not need to be shuffled
            val_loader = Dataloader(val_dataset, batch_size, False, drop_remainder_batch)
        
        config.train_loader = train_loader
        config.val_loader = val_loader
        config.epochs = epochs
        
        return config

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
        """
    Train the model on the provided data.

    Training data may be provided as raw tensors, a Dataset,
    or a DataLoader. When using raw input tensors, labels must
    be provided through the y argument.

    Args:
        data: Training data.
        y: Training labels when using raw input tensors.
        epochs: Number of training epochs.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the training data.
        drop_remainder_batch: Whether to drop incomplete batches.
        validation_split: Fraction of training data reserved for validation.
        validation_data: Separate validation dataset or dataloader.

    Returns:
        A History object containing recorded training metrics.
    """
        config = self._prepare_training_input(
            data,
            y,
            epochs,
            batch_size,
            shuffle,
            drop_remainder_batch,
            validation_split,
            validation_data
        )

        history = self._train(config)
        return history
    
    def _train(self, config) -> History:
        history = History()
        
        for epoch in range(config.epochs):
            metrics = Metrics()

            metrics.loss = self._train_epoch(config, history.epochs)
            if config.val_loader != None:
                metrics.val_loss = self._validate_epoch(config, history.epochs)
            history.epoch_step()
            history.log(metrics)

            CLI.epoch_summary(config, metrics, history.epochs)

        return history
    
    def _train_epoch(self, config, epoch_index):
        epoch_loss = 0

        batch_index = 0

        for batch_x, batch_y in config.train_loader:
            batch_index += 1

            batch_loss = self._fit_batch(batch_x, batch_y)
            epoch_loss += batch_loss

            # update progress bar
            batch_count = config.train_loader.num_batches
            front_str = f"Epoch: {epoch_index + 1} / {config.epochs}"
            end_str = f"Batch {batch_index} / {batch_count} Loss: {batch_loss:.5f}"
            CLI.progress_bar((batch_index / batch_count), 40, front_str, end_str)

        mean_loss = epoch_loss / config.train_loader.size
        return mean_loss
    
    def _validate_epoch(self, config, epoch_index):
        epoch_loss = 0

        batch_index = 0

        for batch_x, batch_y in config.val_loader:
            batch_index += 1

            batch_loss = self._validate_batch(batch_x, batch_y)
            epoch_loss += batch_loss

            # update progress bar
            batch_count = config.val_loader.num_batches
            front_str = f"Epoch: {epoch_index + 1} / {config.epochs}"
            end_str = f"Batch {batch_index} / {batch_count} Validation Loss: {batch_loss:.5f}"
            CLI.progress_bar((batch_index / batch_count), 40, front_str, end_str)

        mean_loss = epoch_loss / config.train_loader.size
        return mean_loss
    
    def _validate_batch(self, batch_x, batch_y):
        with ntk.no_grad():
            output = self.model(batch_x)
            loss = self.loss(output, batch_y)
            return loss.data[0]
    
    def _fit_batch(self, batch_x, batch_y) -> int:
        output = self.model(batch_x)
        loss = self.loss(output, batch_y)
        self.backprop(loss)
        gc.collect()
        return loss.data[0]
    
    def backprop(self, loss:Tensor):
        loss.backward()
        self.optimizer.optimize()
        self.optimizer.clear_grad()