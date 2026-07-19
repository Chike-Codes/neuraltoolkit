# Optimizers
`Optimizer` is the base class that all of NTK's optimizers draw from. 
    An otpimizer is the algorithm that updates a model's weights and biases.
    It determins how to adjust the parameters based on their gradient and the 
    optimizers hyper-parameters

## `optimize()`
Applies the optimizer function to the model parameters

## `clear_grad()`
Applies the optimizer function to the model parameters

# Parameters
All optimizers require the parameters to be optimized be passed as an argument.
All `modules` with learnable parameters can access those parameters using `.paramters()`

Examples:
```python
layer = ntk.Dense(10, 4)

params = layer.parameters
```

```python
model = ntk.Sequential(
    ntk.Dense(10, 4),
    ntk.Relu(),
    ntk.Dense(4, 1)
)

params = model.parameters()
```