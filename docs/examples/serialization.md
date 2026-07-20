# **Serialization**
Saving and loading models with NTK is easy. Simply call save(), on modules like Sequential() or any Layer, and pass the file path. Then call load() on initializaiton to load your saved model.

Below is a quick example demonstrating module serialization. A simple model and dataset are created. The model's outputs are written to the terminal. Then the model is saved and loaded into a new instance. The model's outputs are written to the terminal again to prove the model was correctly serialized.

```python
import neuraltoolkit as ntk

input_data = ntk.Tensor([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9], [1.0]])

model = ntk.Sequential(
    ntk.Dense(in_shape=1, out_shape=10),
    ntk.Sigmoid(),
    ntk.Dense(in_shape=10, out_shape=1)
)

with ntk.no_grad():
    predictions = model(input_data)
    print("First Predictions: ", predictions)

    model.save("model.ntk")
    new_model = ntk.Sequential.load("model.ntk")

    new_predictions = new_model(input_data)
    print("Second Predictions: ", new_predictions)
```

## Terminal Output
Terminal output should look like this. The first and second predictions should be exactly the same.

```terminal
First Predictions:  Tensor:
 [[0.9462638 ]
 [0.9482154 ]
 [0.9501593 ]
 [0.9520917 ]
 [0.95400906]
 [0.9559075 ]
 [0.957784  ]
 [0.95963526]
 [0.9614582 ]
 [0.96325016]] 

Second Predictions:  Tensor:
 [[0.9462638 ]
 [0.9482154 ]
 [0.9501593 ]
 [0.9520917 ]
 [0.95400906]
 [0.9559075 ]
 [0.957784  ]
 [0.95963526]
 [0.9614582 ]
 [0.96325016]] 
```