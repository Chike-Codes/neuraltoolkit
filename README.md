[![PyPI version](https://img.shields.io/pypi/v/ntk-ml?maxAge=0)](https://pypi.org/project/ntk-ml/)

![NTK Banner](https://raw.githubusercontent.com/Chike-Codes/neuraltoolkit/refs/heads/main/NTK%20Banner.png)

# Neural Tool Kit (NTK)
Neural Tool Kit (NTK) is a machine learning framework built from
scratch in Python on top of NumPy. The project aims to provide a
hands-on exploration of the core systems behind modern deep learning
frameworks, including tensors, automatic differentiation, neural
network layers, optimization, and training workflows.

NTK was created as a learning-focused project to explore the
fundamental systems behind modern deep learning frameworks.
While it serves as a personal educational project, it may also
be useful to students, hobbyists, and anyone interested in
understanding how machine learning frameworks work internally.

## Motivation
I built NTK to develop a deeper understanding of the systems that power modern deep learning frameworks. By implementing these components from scratch, I can explore how they work internally rather than treating them as black boxes.

## Features
- Tensor abstraction with automatic differentiation
- Dense, Conv2D, and MaxPool layers
- Built-in activation functions
  - ReLU
  - Sigmoid
  - Tanh
  - Softmax
  - Linear
- Optimizers and loss functions
- Dataset and DataLoader abstractions
- Training utilities and trainer system
- Sequential container module
- Support for custom layers, activations, losses, and optimizers

## Current Limitations
NTK is currently CPU-only and remains under active development.
The following features are planned but not yet available:

- GPU computation
- Additional container modules
- RNN support
- Transformer support
- A polished reinforcement learning API

## Installation
```powershell
python -m pip install ntk-ml
```

## Import
```python
import neuraltoolkit as ntk
```

## Quickstart
```python
import neuraltoolkit as ntk


x = ntk.Tensor(training_data)
y = ntk.Tensor(training_labels)

model = ntk.Sequential(
  ntk.Dense(in_shape=4, out_shape=32),
  ntk.Relu(),
  ntk.Dense(in_shape=32, out_shape=10),
  ntk.Tanh()
)

trainer = ntk.Trainer(
  module=model,
  optimizer=ntk.Adam(parameters=model.parameters(), learning_rate=3e-4),
  loss=ntk.MeanSquaredError()
)

trainer.fit(x, y, epochs=100)

predictions = model(X)
```


## Examples
### XOR
```python
import neuraltoolkit as ntk
import numpy as np
import matplotlib.pyplot as plt

x = ntk.Tensor([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = ntk.Tensor([
    [0],
    [1],
    [1],
    [0]
])

model = ntk.Sequential(
    ntk.Dense(in_shape=2, out_shape=4),
    ntk.Tanh(),
    ntk.Dense(in_shape=4, out_shape=1),
    ntk.Sigmoid()
)

trainer = ntk.Trainer(
    module=model,
    optimizer=ntk.Adam(parameters=model.parameters(), learning_rate=0.01),
    loss=ntk.BinaryCrossEntropy()
)


history = trainer.fit(x, y, epochs=500)
print(model(x))
history.plot("loss")

# ----------------------Visualizing-------------------------

# defining a boundary
x_min, x_max = 0, 1
y_min, y_max = 0, 1

#Creating the grid
step_size = 0.01
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, step_size),
    np.arange(y_min, y_max, step_size)
)

grid_points = np.c_[xx.ravel(), yy.ravel()]
with ntk.no_grad():
    predictions = model(ntk.Tensor(grid_points))

Z = predictions.data.reshape(xx.shape)
print(Z)

# Plotting

plt.figure(figsize=(6, 5))
plt.contourf(xx, yy, Z, alpha=0.8, cmap="coolwarm")
plt.contour(xx, yy, Z, colors='k', levels=[0.5], linewidths=1.5)

plt.title("XOR Decision Boundary")
plt.show()
```

### Mnist Digits (CNN)
```python
import numpy as np
import neuraltoolkit as ntk

model = ntk.Sequential(
    ntk.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0),
    ntk.Relu(),
    ntk.Adaptive_Max_Pool2d(H_out=13, W_out=13),
    ntk.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
    ntk.Relu(),
    ntk.Adaptive_Max_Pool2d(H_out=5, W_out=5),
    ntk.Flatten(),
    ntk.Dense(in_shape=1600, out_shape=128),
    ntk.Relu(),
    ntk.Dense(in_shape=128, out_shape=10)
)

train_dataset, val_dataset = ntk.datasets.mnist()

trainer = ntk.Trainer(
    module=model,
    optimizer=ntk.Adam(parameters=model.parameters(), learning_rate=3e-4),
    loss=ntk.CategoricalCrossEntropy()
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

model.save(".model_conv.ntk")
print("Model Saved!")
```