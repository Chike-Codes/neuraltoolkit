import struct
from array import array
import numpy as np
import matplotlib.pyplot as plt

testLabelsPath = "./archive/t10k-labels.idx1-ubyte"
testImagesPath = "./archive/t10k-images.idx3-ubyte"
trainLabelsPath = "./archive/train-labels.idx1-ubyte"
trainImagesPath = "./archive/train-images.idx3-ubyte"

def load_images_labels(imagesPath, labelsPath):
    with open(labelsPath, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError(f"Magic number expected a value of 2049, got {magic}")
        labels = array("B", file.read())

    with open(imagesPath, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError(f"Magic number expected a value of 2051, got {magic}")
        imageData = array("B", file.read())

    images = [[0] * rows * cols] * size
    for i in range(size):
        img = np.array(imageData[i * rows * cols : (i + 1) * rows * cols])
        img = img.reshape(rows, cols)
        images[i] = img
    
    return images, labels