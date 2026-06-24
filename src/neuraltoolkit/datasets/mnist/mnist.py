from neuraltoolkit.core.tensor import Tensor
from neuraltoolkit.data.dataset import Dataset
from neuraltoolkit.datasets.mnist import mnistloader
from pathlib import Path
import numpy as np


def mnist():
    """
    Returns a training Dataset and a validation Dataset

    # Dataset Shapes:
        ## Training Data:
            x: (60000, 1, 28, 28)
            y: (60000, 10)

        ## Validation Data:
            x: (10000, 1, 28, 28)
            y: (10000, 10)
    """
    train_dataset = MNIST(train=True)
    val_dataset = MNIST(train=False)
    return train_dataset, val_dataset

def MNIST(train:bool):
    """
    Returns a training Dataset or a validation Dataset

    Args:
        train (bool): returns training set if **True** or validation set if **False**

    # Dataset Shapes:
        ## Training Data:
            x: (60000, 1, 28, 28)
            y: (60000, 10)

        ## Validation Data:
            x: (10000, 1, 28, 28)
            y: (10000, 10)
    """
    path = Path(__file__).parent / "archive"
    
    if train:
        labels_path = path / "train-labels.idx1-ubyte"
        images_path = path / "train-images.idx3-ubyte"
    else:
        labels_path = path / "t10k-labels.idx1-ubyte"
        images_path = path / "t10k-images.idx3-ubyte"

    images, labels = mnistloader.load_images_labels(images_path, labels_path)

    data = np.array(images, dtype=np.float32).reshape(len(images), 1, 28, 28)
    data = data / 255
    labels = np.eye(10)[labels]

    data = Tensor(data)
    labels = Tensor(labels)
    
    return Dataset(data, labels)



