# **Conv2D**
`Conv2d` is the standard layer for image processing within a neural network

This layer performs a convolution by sliding a filter (The kernel) across the image. A new, scaled down image is output, calculated by the weighted sum of the pixels under the kernel at each position as well as adding a bias term.

The `out_channels` parameter determins the number of learned features the model will extract from the image. 

## Output Dimensions
The dimensions of the output image is determined by the values for `kernel_size`, `stride`, and `padding`
The formula for each dimension is the following

$$
Output\ Size = \left \lfloor \frac{input\ size + 2\times padding - kernel\ size}{stride} \right \rfloor + 1
$$

::: neuraltoolkit.modules.layers.conv2d.Conv2d

## implementation

```python
import neuraltoolkit as ntk
import numpy as np

image_count = 100
in_channels = 3
height = 28
width = 28

x = ntk.Tensor(
    np.random.rand(image_count, in_channels, height, width)
)

layer = ntk.Conv2d(
    in_channels=in_channels,
    out_channels=32,
    kernel_size=2,
    stride=1,
    padding=0
)

y = layer(x)
```