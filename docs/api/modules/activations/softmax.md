# **Softmax**

::: neuraltoolkit.modules.activations.softmax.Softmax

```python
ntk.Softmax()
```

The `Softmax` activation function takes a raw, unnormalized vector of model outputs and transforms them into a probality distribution (All outputs add up to 1.0)

## The Module Vs. The Function
NTK has two version of this activation function, `Softmax()` and `softmax()`. The first is the layer/module version used directly as a part of model. The second is just a function that transforms incoming data. The main difference is that `Softmax()` requires an instance and `softmax()` does not. The reason for this is that oftentimes a model using softmax won't include it during training because `CrossEntropyLoss` has softmax built in and expects raw data. Usually, after training, models will just have their output passed through `softmax()` or the can be restructured to have the module incorporated into the architecture.