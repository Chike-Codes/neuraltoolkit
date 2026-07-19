# **Max Pool 2D**
`Max_Pool2d` is an image processing layer, used for downscaling images while retaining feature data

The `Max_Pool2d` layer has no learnable perameters. It applies an opperation similar to a convolution, by sliding a window over images. The resulting downscaled images are made up of the max values from each window position.

## Output Dimensions
The dimensions of the output image are calculated the same as for `Conv2d` [Referenced Here](conv2d.md#output-dimensions)

::: neuraltoolkit.modules.layers.max_pool2d.Max_Pool2d

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

layer = ntk.Max_Pool2d(
    kernel_size=3,
    stride=2,
    padding=0
)

y = layer(x)
```