import torch
from torch import nn
import matplotlib.pyplot as plt

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

def plot_predictions(train_data=X_train.cpu(),
                     train_labels=y_train.cpu(),
                     test_data=X_test.cpu(),
                     test_labels=y_test.cpu(),
                     predictions=None):
    """
    Plots training data, test data and compares predictions if not None
    """
    plt.figure(figsize=(10, 7))

    plt.scatter(train_data, train_labels, label=f"Training data", c="b")
    plt.scatter(test_data, test_labels, label=f"Test data", c="g")
    if predictions is not None:
        plt.scatter(test_data, predictions, label="Predictions", c="r")

    plt.legend()
    plt.grid()
    plt.show()


class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        # Use nn.Linear instead of defining weights and biases manually
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

torch.manual_seed(42)
model_1 = LinearRegressionModelV2()
print(model_1.state_dict())
model_1.to(device)

# 2. Train model
# loss function
loss_function = nn.L1Loss()

# optimizer
optimizer = torch.optim.SGD(model_1.parameters(), lr=0.001)

# training loop
epochs = 2000
for epoch in range(epochs):
    model_1.train()                         # set training mode
    y_pred = model_1(X_train)               # forward pass
    loss = loss_function(y_pred, y_train)   # compute loss
    optimizer.zero_grad()                   # reset optimizer (clear gradients)
    loss.backward()                         # perform backward pass
    optimizer.step()                        # update weights
    
    # Testing
    model_1.eval()
    with torch.inference_mode():
        test_predictions = model_1(X_test)
        test_loss = loss_function(test_predictions, y_test)

    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{epochs}], Training Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")


plot_predictions()
model_1.eval()
with torch.inference_mode():
    predictions = model_1(X_test)
print(model_1.state_dict())
plot_predictions(predictions=predictions.cpu())

# Save model
torch.save(model_1.state_dict(), "model_1.pth")
loaded_state = torch.load("model_1.pth")
model_2 = LinearRegressionModelV2()
model_2.load_state_dict(loaded_state)

print(model_2.state_dict())
model_2.to(device)
model_2.eval()
with torch.inference_mode():
    loaded_predictions = model_2(X_test)
print(loaded_predictions == predictions)
