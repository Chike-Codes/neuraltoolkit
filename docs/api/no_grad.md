`no_grad()` is tensor state function that controls whehter gradient chain graphs are created.

use `with ntk.no_grad():` to enter no gradient mode. No gradient mode is meant to be used post training when tracking gradients is not necessray.

```python
with ntk.no_grad():
    model(data)
```