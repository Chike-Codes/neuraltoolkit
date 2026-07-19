Serialization for native NTK modules is simple. Just call `save()` on the module you'd like to save and provide the desired path. It's recomended that you're saved modules have the `.ntk` file extension, for clarity. When loading your module call `load()` with the file path when instantiating a new instance of the same module type.

## Example
```python
import neuraltoolkit as ntk

model = ntk.Sequential(
    ntk.Dense(in_shape=2, out_shape=4),
    ntk.Tanh(),
    ntk.Dense(in_shape=4, out_shape=1),
    ntk.Sigmoid()
)

model.save("my_model.ntk")
```