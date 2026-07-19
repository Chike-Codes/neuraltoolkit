# **Losses**

## Mean Squared Error
```python
ntk.MeanSquaredError()
```
`Mean Squared Error` (MSE) is a standard loss metric for regression models. MSE measures the average squared difference between the predicted and actual values. By squaring the differences, the negative and positive values do not cancel out, and models are penalized more heavily for large errors or outliers.

## Binary Cross Entropy
```python
ntk.BinaryCrossEntropy()
```
`Binary Cross Entropy` (BCE or Log Loss) is a loss metric for models that output a probality value between 1 and 0. It heavily penalizes confident, incorrect predictions. Binary Cross Entropy is the standard loss metric for binary classification tasks.

## Cross Entropy
```python
ntk.CrossEntropy()
```
`Cross Entropy` (CE) is the standard loss metric for multi-class classification as a probability distribution.

In NTK `Cross Entropy` applies a `softmax()` function to the data that flows through it and expects the output data from the model to be linear. To get a probility distribution from the model during deployment, plug the outputs of the model into a `Softmax` function
```python
# Training a classification model to convert binary to 0-4
import neuraltoolkit as ntk

x = ntk.Tensor([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = ntk.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

model = ntk.Sequential(
    ntk.Dense(in_shape=2, out_shape=10),
    ntk.Sigmoid(),
    ntk.Dense(in_shape=10, out_shape=4)
)

# Training the model
trainer = ntk.Trainer(
    module=model,
    optimizer=ntk.Adam(parameters=model.parameters(), learning_rate=3e-4),
    loss=ntk.CrossEntropy()
)

trainer.fit(x, y, epochs=100)

# Testing the model
predictions = model(x)
probabilities = ntk.softmax(predictions)
print("probabilities \n", probabilities)
```