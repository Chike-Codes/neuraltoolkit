`Modules` are the core components that make up deep learning models. **Layers**, **Activation Functions**, and **containers** are all modules. all modules share the common traits that they take a batch of data and output a batch of transformed data, and that they can be chained together.

## Layers
Layers are the most common type of module and what most think of as deep learning components. All layers have learnable parameters (`weights` and `biases`).

## Activation Functions
Activations are mathematical functions applied after layers and determin final neuron output. Activation functions allow models to learn complex non-linear representations, and prevent collapse during training.

## Containers
containers are orchestrator modules. They take other modules and define how data flows through the model.

## Modules
| Name                | Module Type         |
|---------------------|---------------------|
| [Dense](layers/dense.md)               | Layer               |
| [Conv2d](layers/conv2d.md)              | Layer               |
| [Flatten](layers/flatten.md)             | Layer               |
| [Max Pool2d](layers/max_pool2d.md)          | Layer               |
| [Adaptive Max Pool2d](layers/adaptive_max_pool2d.md) | Layer               |
| [Sigmoid](activations/sigmoid.md)             | Activation Function |
| [Tanh](activations/tanh.md)                | Activation Function |
| [ReLU](activations/relu.md)                | Activation Function |
| [Leaky ReLU](activations/leaky_relu.md)          | Activation Function |
| [Softmax](activations/softmax.md)             | Activation Function |
| [Sequential](containers/sequential.md)          | container           |
