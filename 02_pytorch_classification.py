from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from helpers import plot_decision_boundary, train_and_test, accuracy
from circlemodel import CircleModelV1, CircleModelV2, CircleModelV3

random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device: ", device)

# Create 100 circle samples for binary classification
n_samples = 1000

# Toy dataset for playing around: two circles
X, y = make_circles(n_samples, noise=0.03, random_state=random_seed)

# X contains the x and y coordinates of data points, y contains the labels (1 or 0)

print(len(X), len(y))
print(X[:5], y[:5])

circles = pd.DataFrame({
    "X1": X[:, 0],
    "X2": X[:, 1],
    "label": y}
    )
print(circles.head)

# plt.scatter(x=X[:, 0], y=X[:, 1], c=y, cmap=plt.cm.RdYlBu)
# plt.show()

# View first sample
X_sample = X[0]
y_sample = y[0]
print(f"Sample values: X: {X_sample}, y: {y_sample}")
print(f"Sample shapes: X: {X_sample.shape}, y: {y_sample.shape}")

# Turn data into tensors on the selected device
X = torch.from_numpy(X).float().to(device)
y = torch.from_numpy(y).float().to(device)
print(X[:5], y[:5])

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2, # 20% of samples will be the test set
                                                    random_state=random_seed)
print(len(X_train), len(X_test), len(y_train), len(y_test))

# Building a model
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()

        # Rule of thumb: the more hidden features, the more opportunity our model has to learn patterns in the data
        layer_size = 5
        self.layer_1 = nn.Linear(in_features=2, out_features=layer_size) # hidden layer
        self.layer_2 = nn.Linear(in_features=layer_size, out_features=1) # output layer

    # Dfines the forward pass of the model
    def forward(self, x):
        self.layer_2(self.layer_1(x)) # x -> layer 1 -> layer 2 -> output

model_0 = CircleModelV0()
model_0.to(device)

# Use nn.Sequential to define the model. Shortcut for straight sequential feedforward neural networks
model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
).to(device)
print(model_0.state_dict())
# Make predictions with the model
with torch.inference_mode():
    untrained_preds = model_0(X_test)
print(untrained_preds[:10])

# Which loss function to use?
# - for regression (predicting a number): MAE, MSE (typically)
# - for classification (predicting a label): binary cross-entropy, categorical cross-entropy (cross-entropy loss) (typically)
loss_fn = nn.BCEWithLogitsLoss() # built-in sigmoid activation function, more numericaly stable than using BCELoss and separate sigmoid

# What optimizer to use?
# - for regression and classification: Adam, SGD (typically)
optimizer = torch.optim.SGD(model_0.parameters(), lr=0.1)

# Training the model
def transform_output(y_pred):
    return torch.round(torch.sigmoid(y_pred))

train_and_test(epochs=100, model=model_0, loss_fn=loss_fn, transformation=transform_output, optimizer=optimizer, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

# Visualize model 0
fig = plt.figure(figsize=(12, 6))
fig.suptitle("Model 0")
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_0, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_0, X_test, y_test)

# The model is not learning anything. The decision boundary is a straight line, which is not what we want.
# How de we improve this? -> HYPERPARAMETER TUNING
# - Add more layers
# - Add more hidden units
# - Train for longer
# - Change the activation functions
# - Change the learning rate
# - Change the loss function
# - Change the optimizer
# - (Change the data)

# see circlemodel.py for adjusted model code
model_1 = CircleModelV3().to(device)

train_and_test(
    epochs=5000,
    model=model_1,
    loss_fn=loss_fn,
    transformation=transform_output,
    optimizer=torch.optim.SGD(model_1.parameters(), lr=0.1),
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    metrics={"accuracy": accuracy}
    )
# Visualize model 1
fig = plt.figure(figsize=(12, 6))
fig.suptitle("Model 1")
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_1, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_1, X_test, y_test)
plt.show()

## The missing piece: non-linearities