import numpy as np

class Dataloader:
    """
    Iterates over a dataset in mini-batches.

    The DataLoader handles batching and optional shuffling,
    providing an efficient interface for training models.

    Args:
        dataset (Dataset): Dataset to iterate over.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset at the
            beginning of each epoch. (defaults to False)
        drop_remainder (bool): determins if the final mini batch should be dropped
            if the data cannot be segmented evenly. (defaults to False)
        
    """
    def __init__(self, dataset, batch_size=None, shuffle=False, drop_remainder=False):
        self.dataset = None
        self._indices = None

        self.set_dataset(dataset)

        if batch_size == None:
            self.set_batch_size(self.dataset.size)
        else:
            self.set_batch_size(batch_size)

        # Flags
        self.drop_remainder = drop_remainder
        self._shuffle_flag = shuffle

    def __iter__(self):
        self.index = 0

        if self._shuffle_flag:
            self._shuffle()

        return self
    
    def __next__(self):
        if self.index >= self.dataset.size:
            raise StopIteration
        
        batch_slice = slice(self.index, self.index + self.batch_size)
        batch_indices = self._indices[batch_slice]

        if batch_indices.shape[0] != self.batch_size and self.drop_remainder:
            raise StopIteration
        
        x = self.dataset.x[batch_indices]
        y = self.dataset.y[batch_indices]

        self.index += self.batch_size

        return (x, y)

    def set_dataset(self, dataset):
        self.dataset = dataset
        self._indices = np.arange(self.dataset.size)

    def shuffle(self, flag:bool):
        self._shuffle_flag = flag

    def _shuffle(self):
        rng = np.random.default_rng()
        
        #in-place shuffle operation
        rng.shuffle(self._indices)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def split(self, frac:float, shuffle:bool=False):
        """Returns a training and validation Dataloader"""
        train_subset, val_subset = self.dataset.split(frac, shuffle)
        train_loader = Dataloader(train_subset, self.batch_size, self._shuffle_flag, self.drop_remainder)
        val_loader = Dataloader(val_subset, self.batch_size, shuffle=False, drop_remainder=self.drop_remainder)

        return train_loader, val_loader

    @property
    def size(self):
        return self.dataset.size
    
    @property
    def num_batches(self):
        result = self.size // self.batch_size
        result = result + 1 if result % 1 != 0 else result
        return result