`Sequential` is a container module. It takes other modules as arguments and defines how data flows through them.
This container defines standard feed-forward architecture. Data flows from each layer in order.

## Initializing a model
Model archicture can be defined through either initialization arguments or with `.add_layer()`

```python
model = ntk.Sequential(
    ntk.Dense(10, 32),
    ntk.Sigmoid(),
    ntk.Dense(32, 64),
    ntk.sigmoid(),
    ntk.Dense(64, 10)
)
```

```python
model = ntk.Sequential()
model.add_layer(ntk.Dense(10, 32))
model.add_layer(ntk.Sigmoid())
model.add_layer(ntk.Dense(32, 64))
model.add_layer(ntk.Sigmoid())
model.add_layer(ntk.Dense(64, 10))
```