# **Tensor**
::: neuraltoolkit.core.tensor.Tensor

## **Data**
---
`Tensors` hold multi-dimensional arrays of numeric data only. If the data passed to `Tensor` isn't a NumPy array, it will try to convert it to one and throw an error if not possible. The data types accepted are `NumPy arrays`, python `lists` and `tuples`, and `numerics`.

**Note** that no matter the datatype passed `Tensor` will always store it as a NumPy array.
```python
import neuraltoolkit as ntk
import numpy as np

a = ntk.Tensor(np.array([1, 2, 3, 4, 5]))
# array([1, 2, 3, 4, 5])

b = ntk.Tensor([0.1, 0.2, 0.3, 0.4, 0.5])
# array([0.1, 0.2, 0.3, 0.4, 0.5])

c = ntk.Tensor((0, 0, 0, 0, 0))
# array([0, 0, 0, 0, 0])

d = ntk.Tensor(10)
# array([10])
```

### Accessing Tensor Data
Tensor data can be accessed via `Tensor.data`

To display tensor data to the terminal simply use `print()`:
```python
a = ntk.Tensor([1, 2, 3, 4, 5])
print(a)
```
```terminal
Terminal Output

Tensor:
 [1. 2. 3. 4. 5.] 
```

## **Gradients**
---
`Tensors` can hold gradients. Gradients are the values that are updated during backpropagation and are used to update the parameters of a model. To create a tensor with gradients set the `requires_grad` argument to `True` upon instantiation. If not defined then it will default to `False`
```python
ntk.Tensor([1, 2, 3], requires_grad=True)
```

When defining a tensor as the parameters to a model, use [Parameter](parameter.md) A subclass of `Tensor` that always uses gradients

To access a tensor's gradient use `tensor.grad`

Tensor gradients can be cleared using `.clear_grad()`. If training a model without [Trainer](trainer.md), the gradients must be cleared after each optimization step.

## **Operations**
---
Beyond holding data, `Tensors` can perform math on their data and provide functionality for slicing and indexing similar to arrays and lists.
### Math
Tensors accept most mathematical operators with functionality for tensor to tensor and tensor to scalar math.

All tensor math operaitons are not in-place and return a new tensor

| Operation             | Expression | Result                      |
| --------------------- | ---------- | --------------------------- |
| Addition              | `a + b`    | Element-wise addition       |
| Subtraction           | `a - b`    | Element-wise subtraction    |
| Multiplication        | `a * b`    | Element-wise multiplication |
| Division              | `a / b`    | Element-wise division       |
| Matrix multiplication | `a @ b`    | Matrix multiplication       |
| Power                 | `a ** 2`   | Element-wise exponentiation |
| Negation              | `-a`       | Element-wise negation       |

```python
x = ntk.Tensor([1, 2, 3])
y = ntk.Tensor([4, 5, 6])

z = (x + y) * 2
# z -> Tensor: [10, 14, 18]
```

### Indexing
`Tensors` can be indexed, sliced, and interated over similar to numpy arrays.

| Method    | Syntax              |
|-----------|---------------------|
| Indexing  | tensor[a]           |
| Slicing   | tensor[a:b:c]       |
| Iteration | for item in tensor: |

## **Autograd**
---
`Tensors` have a built-in autograd (automatic gradient differentiation) engine making backpropagation simple.

Whenever an operation is applied to a tensor, whether it be indexing or math, a gradient chain graph is built. Simply calling `backward()` on the final tensor (usually loss) will automatically update the gradients of all tensors in the chain, after which optimizers can be used to update parameters.

```python
a = ntk.Tensor([1, 2], requires_grad=True)
b = ntk.Tensor([3, 4], requires_grad=True)

c = a * b

z = c - 3

z.backward()

print(a.grad) # [3, 4]
print(b.grad) # [1, 2]
```