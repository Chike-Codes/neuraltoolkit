# **MNIST**
After successfully training a model on XOR, the MNIST hand-written digits dataset is a nice step up in complexity. Like the XOR example, this is a plug-and-play script that can be copied into any Python environment. This example covers the process of creating and implementing a convolutional neural network (CNN). 

The code below demonstrates the added complexity of image processing, as well as more advanced NTK features and abstractions. This example displays the Conv2d(), Flatten(), and Adaptive_Max_Pool2d() layers for image processing. It includes the use of the NTK's included MNIST dataset, allowing for the datasets to be downloaded and automatically loaded into a Dataloader object, making managing how the data is loaded and distributed simplified and automated.

```python
import numpy as np
import neuraltoolkit as ntk

model = ntk.Sequential(
    ntk.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0),
    ntk.Relu(),
    ntk.Adaptive_Max_Pool2d(height=13, width=13),
    ntk.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
    ntk.Relu(),
    ntk.Adaptive_Max_Pool2d(height=5, width=5),
    ntk.Flatten(),
    ntk.Dense(in_shape=1600, out_shape=128),
    ntk.Relu(),
    ntk.Dense(in_shape=128, out_shape=10)
)

train_dataset, val_dataset = ntk.datasets.mnist()

trainer = ntk.Trainer(
    module=model,
    optimizer=ntk.Adam(parameters=model.parameters(), learning_rate=3e-4),
    loss=ntk.CrossEntropy()
)

history = trainer.fit(
    data=train_dataset,
    epochs=1,
    validation_data=val_dataset,
    batch_size=32,
    shuffle=True
)

with ntk.no_grad():
    predictions = model(val_dataset.x)

predictions_argmax = np.argmax(predictions.data, axis=-1)
labels_argmax = np.argmax(val_dataset.y.data, axis=-1)

percentage = np.mean(predictions_argmax == labels_argmax) * 100
print(f"Test Accuracy: {percentage}%")
```