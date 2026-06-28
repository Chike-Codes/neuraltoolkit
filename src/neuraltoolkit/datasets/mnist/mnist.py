from neuraltoolkit.core.tensor import Tensor
from neuraltoolkit.data.dataset import Dataset
from neuraltoolkit.datasets.management.data_resource import DatasetResource
from neuraltoolkit.datasets.management.retrieve import retrieve_path
from neuraltoolkit.datasets.mnist import mnistloader
from pathlib import Path
import numpy as np


MNIST_RESOURCE = DatasetResource(
    name="mnist",
    url="https://github.com/Chike-Codes/neuraltoolkit/releases/download/datasets-v1/mnist.npz",
    sha256="4dc5a9aaba44b3fde1d478c7c21f18bcfc67e34f139fe072b43d7ad5848365b2",
    file_name="mnist.npz"
)

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
    path = retrieve_path(MNIST_RESOURCE)
    with np.load(path) as cache:
        if train:
            data = cache["train_x"]
            labels = cache["train_y"]
        else:
            data = cache["val_x"]
            labels = cache["val_y"]

    data = Tensor(data)
    labels = Tensor(labels)
    
    return Dataset(data, labels)



