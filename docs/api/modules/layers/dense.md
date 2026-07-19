# **Dense**
`Dense` is the standard fully connected layer used in most typical neural networks.

Each output feature is computed as the weighted sum of all input features, followed by a bias term.
::: neuraltoolkit.modules.layers.dense.Dense

## Example 

```python
import neuraltoolkit as ntk
import numpy as np

sample_count = 50
input_features = 10

x = ntk.Tensor(
    np.random.rand(sample_count, input_features)
)

layer = ntk.Dense(
    in_shape=input_features,
    out_shape=4
)

y = layer(x)
```