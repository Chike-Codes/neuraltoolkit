# **Adaptive Max Pool 2D**
`Adapctive_Max_Pool2d` functions the same as `Max_Pool2d` but rather than defining the kernel, stride, and padding, the desired otuput dimensions are defined and the rest is calculated automatically.

When defining the output dimensions, it's important to understand what the resulting stride and kernel will be.
Kernel and Stride, per axis, are calculated as follows:

$$
Stride =  \left\lfloor \frac{input\ size} {output\ size} \right\rfloor
$$

$$
Kernel = input\ size - (output\ size - 1) \times Stride
$$

::: neuraltoolkit.modules.layers.adaptive_max_pool2d.Adaptive_Max_Pool2d

## Implementation
```python
import neuraltoolkit as ntk
import numpy as np

sample_count = 100
channels = 3
height = 28
width = 28

x = ntk.Tensor(
    np.random.rand(sample_count, channels, height, width)
)

# Stride = (2, 2)
# Kernel = (4, 4)

layer = ntk.Max_Pool2d(
    height=13
    width=13
)

y = layer(x)
```