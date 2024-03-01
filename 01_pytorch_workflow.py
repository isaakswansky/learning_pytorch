# PyTorch Workflow by end-to-end example

what_were_covering = {
    1: "data (prepare and load)",
    2: "build model",
    3: "train model",
    4: "making predictions (inference) and evaluating model",
    5: "saving and loading a model",
    6: "putting it all together"
}
print(what_were_covering)

import torch
from torch import nn # nn contains all building blocks for creating neural networks
import matplotlib.pyplot as plt

# check pytorch version
print(torch.__version__)

# 1. Data preparation and loading
# ML: Part1: Turn data into numbers, Part2: Build model to learn patterns in numbers

# Create som dat using the linear regression: Y = a + bX
# b = weight, a = bias

weight = 0.7
bias = 0.3

start = 0.
end = 1.
step = 0.02
X = torch.arange(start, end, step).unsqueeze(1) # unsqueeze adds an extra dimension to tensor
y = weight * X + bias

for x_val, y_val in zip(X, y):
    print(f"{x_val.item():.2f} -> {y_val.item():.4f}")

# Split data into training, validation and test sets IMPORTANT!!
# 1) Training set: used to train the model (~60%-80%)
# 2) (optional) Validation set: used to evaluate and tune the model (~10-20%)
# 3) Test set: used to test the model (~10-20%)
    
# Split data into training and test sets (leave validation set out for now)
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print(f"training samples: {len(X_train)} , test samples: {len(X_test)}")

def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
    """
    Plots training data, test data and compares predictions if not None
    """
    plt.figure(figsize=(10, 7))

    plt.scatter(X_train, y_train, label=f"Training data", c="b")
    plt.scatter(X_test, y_test, label=f"Test data", c="g")
    if predictions is not None:
        plt.scatter(test_data, predictions, label="Predictions", c="r")

    plt.legend()
    plt.grid()
    plt.show()

# 2. Build model
# Create a linear regression model class
class LinearRegressionModel(nn.Module): # inherit from nn.Module -> almost everything in PyTorch inherits from nn.Module
    def __init__(self) -> None:
        super().__init__()

        # initialize model parameters
        self.weights = nn.Parameter(
            torch.randn(
                1,
                requires_grad=True, # PyTorch will track the gradients for use with torch.autograd
                dtype=torch.float32)
            )
        self.bias = nn.Parameter(
            torch.randn(
                1,
                requires_grad=True,
                dtype=torch.float32)
            )

    # used to compute output of the model (should be overridden in subclasses of nn.Module)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weights + self.bias # this is the linear regression equation
    
# What our model does:
# - start with random values (weight and bias)
# - pass input data through model
# - compare model's predictions with actual targets
# - adjust weights and biases to reduce difference between predictions and targets
# - continue until the difference is as low as possible
    
# Algorithms:
# - Gradient descent
# - Backpropagation
    
# PyTorch model building essentials
# - torch.nn building blocks for computational graphs (like neural networks)
# - torch.nn.Parameter: what parameters should our model try and learn?
# - torch.nn.Module: base class for all neural network modules
# - torch.optim: optimization algorithms (like gradient descent)
# - def forward() - all nn.Module subclasses require a forward method, implements the forward computation of the model
# - torch.utils.data.Dataset: an abstract class representing a dataset (map between label/sample pairs of data)
# - torch.utils.data.DataLoader: wraps a dataset and provides access to underlying data (shuffles data, batches data, etc.)

torch.manual_seed(42) # set random seed for reproducibility
model_0 = LinearRegressionModel()
print(list(model_0.parameters()))
print(model_0.state_dict())

# Making predictions with the model using torch.inference_mode()
# when we pass data to our model, it runs the forward method
with torch.inference_mode(): # turns off gradient tracking -> faster predictions
    y_preds_initial = model_0(X_test) # pretty bad predictions since weights and biases are random

# 3. Train model
# How do we measure how good our model is? -> We need a loss/cost function/criterion
# - loss function: measures how far off model's predictions are from the actual target, lower is better
# How do we improve our model? -> We need an optimizer
# - optimizer: adjusts model's parameters (weight and bias) to reduce the loss

# PyTorch specifics:
# - training loop
# - testing loop
    
# Setup loss and optimizer
loss_fn = nn.L1Loss() # L1 loss (mean absolute error)
optimizer = torch.optim.SGD(model_0.parameters(), lr=0.01) # lr = learning rate (possibly the most important parameter) # HYPERPARAMETER
# the smaller the learning rate, the slower the model learns

# Training loop (and testing loop)
# We need:
# 0. Loop through the data and do the following for each epoch
# 1. forward pass to compute prediction, also called forward propagation
# 2. calculate loss (compare predictions to actual targets)
# 3. Optimizer zero gradients
# 4. Loss backward - move backwards through the network to calculate gradients of the parameters with respect to the loss (backpropagation)
# 5. Optimizer step - use the optimizer to adjust the parameters to reduce the loss (gradient descent)

# An epoch is one loop through the entire dataset
epochs = 1000 # HYPERPARAMETER

# 0. Loop through the data
for epoch in range(epochs):
    print("epoch:", epoch + 1)
    model_0.train() # set model to training mode, only necessary if model contains layers which behave differently in training and testing phase (e.g. activates gradients)
    
    # 1. forward pass
    y_preds = model_0(X_train)

    # 2. calculate loss
    loss = loss_fn(y_preds, y_train)
    print("loss:", loss.item())

    # 3. Optimizer zero gradients
    optimizer.zero_grad()

    # 4. Backpropagation
    loss.backward() # calculate gradients of loss with respect to model parameters

    # 5. Optimizer step
    optimizer.step() # by default, what the optimizer does accumulates over the loops, so we have to zero them above (step 3)
    #print(model_0.state_dict())

    # Testing
    model_0.eval() # turn off different settings not needed for evaluation/testing

    if (epoch + 1) % 10 == 0:
        with torch.inference_mode(): # turn off gradient tracking (& other parts not needed for evaluation)
            # 1. forward pass
            y_preds_test = model_0(X_test)
            # 2. calculate loss
            loss_test = loss_fn(y_preds_test, y_test)
            print(f"test loss: {loss_test.item():.4f}")

plot_predictions(predictions=y_preds_initial)
with torch.inference_mode():
    y_preds = model_0(X_test)
    plot_predictions(predictions=y_preds)