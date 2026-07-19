# **Flatten**
`Flatten` is a standard image processing layer applied in most convolutional neural networks. The Flatten layer has no learnable parameters. it simply applies an opperation to the data that passes through it.

::: neuraltoolkit.modules.layers.flatten.Flatten

## Implementation
```python
import neuraltoolkit as ntk

x = ntk.Tensor([
    [1, 2],
    [3, 4]
])

layer = ntk.Flatten()
y = layer(x) # -> [1, 2, 3, 4]
```