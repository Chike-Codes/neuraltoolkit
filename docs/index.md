[![PyPI version](https://img.shields.io/pypi/v/ntk-ml?maxAge=0)](https://pypi.org/project/ntk-ml/)

![Alt text for accessibility](images/NTK%20Banner.png)

# **Neural Tool Kit (NTK)**
Neural Tool Kit (NTK) is a NumPy-based machine learning framework built from scratch in Python. The project aims to provide a
hands-on exploration of the core systems behind modern deep learning
frameworks, including tensors, automatic differentiation, neural
network layers, optimization, and training workflows.

I designed NTK as a learning-focused project to explore the
fundamental systems behind modern deep learning frameworks.
While it serves as a personal educational project, it may also
be useful to students, hobbyists, and anyone interested in
understanding how machine learning frameworks work internally.

## Motivation
I built NTK to develop a deeper understanding of the systems that power modern deep learning frameworks. By implementing these components from scratch, I can explore how they work internally rather than treating them as black boxes.

## Installation
```powershell
pip install ntk-ml
```

## Quick Start
```python
import neuraltoolkit as ntk

x = ntk.Tensor([[1, 2], [3, 4]])

print(x)

model = ntk.Sequential(
    ntk.Dense(2, 4),
    ntk.Relu(),
    ntk.Dense(4, 1)
)
```