import torch
from torch import nn
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from helpers import plot_predictions, train_and_test
from circlemodel import LinearModelV1

# Check pytorch version
print("pytorch version:", torch.__version__)

# Create device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# 1. Create data for y = ax + b
weight = 0.7
bias = 0.5
start = 0.0
end = 1.0
step = 0.02
x = torch.arange(start, end, step, device=device).unsqueeze(1) # unsqueeze to make it a 2D tensor

y = weight * x + bias
print("x:", x)
print("y:", y)

train_split = int(0.8 * len(x))
X_train, y_train = x[:train_split], y[:train_split]
X_test, y_test = x[train_split:], y[train_split:]

torch.manual_seed(42)
model_1 = LinearModelV1()
print(model_1.state_dict())
model_1.to(device)

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model_1.parameters(), lr=0.001)

train_and_test(
    epochs=2000,
    model=model_1,
    loss_fn=loss_fn,
    optimizer=optimizer,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test
)

model_1.eval()
with torch.inference_mode():
    y_pred = model_1(X_test)

# Visualize model 1
plot_predictions(
    train_data=X_train.cpu(),
    train_labels=y_train.cpu(),
    test_data=X_test.cpu(),
    test_labels=y_test.cpu(),
    predictions=y_pred.cpu()
)
plt.show()

