import numpy as np
from ..core import Tensor

def shuffle_data(x, seed=None):
    rng = np.random.default_rng(seed)
    rng.shuffle(x, axis=0)
    return x


def create_batches(x:Tensor, batch_size:int) -> list:
    count = x.shape[0] // batch_size
    if count == 0:
        count = 1
    batches = np.array_split(x.data, count)
    tensors = []
    for batch in batches:
        tensors.append(Tensor(batch))

    return tensors

    

def split_validation_data(validation_data, validation_split, validation_batch_size):
    if validation_split > 0:
        if validation_split > 1:
            raise ValueError("validation_data must be a fraction between 0 and 1")
        element_count = int(sample_count * validation_split)
        selection = slice(sample_count - element_count, sample_count)
        sample_count -= element_count
        validation_data = (create_batches(x[selection], validation_batch_size), 
                            create_batches(y[selection], validation_batch_size))
        x = np.delete(x, selection, axis=0)
        y = np.delete(y, selection, axis=0)
    elif validation_data != None:
        validation_data = (create_batches(validation_data[0], validation_batch_size), 
                            create_batches(validation_data[1], validation_batch_size))
    return validation_data