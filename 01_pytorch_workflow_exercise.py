import torch
from torch import nn
import matplotlib.pyplot as plt

# device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device", device)

# make it repoducible
torch.manual_seed(42)

# 1. data
start = 0.0
end = 2.0
step = 0.01
weight = 0.3
bias = 0.9
X_data = torch.arange(start, end, step, device=device).unsqueeze(1) # create data on the selected device
Y_data = weight * X_data + bias

training_split = int(0.8 * len(X_data))
X_training, Y_training = X_data[:training_split], Y_data[:training_split]
X_test, Y_test = X_data[training_split:], Y_data[training_split:]

def plot_data(training_data=X_training.cpu(), training_labels=Y_training.cpu(), test_data=X_test.cpu(), test_labels=Y_test.cpu(), predictions=None):
    plt.figure()
    plt.scatter(training_data, training_labels, label="Training Data", c="b")
    plt.scatter(test_data, test_labels, label="Test Data", c="g")
    if predictions is not None:
        plt.scatter(test_data, predictions, label="Prediction", c="r")
    plt.grid()
    plt.show()

plot_data()

class LinearRegressionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        return self.linear_layer(x)
    
model = LinearRegressionModel()
model.to(device)
print(model.state_dict())

loss_function = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

losses = []
test_losses = []
test_epochs = []
epochs = range(300)
for epoch in epochs:
    model.train()
    predictions = model(X_training)
    loss = loss_function(predictions, Y_training)
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        model.eval()
        test_epochs.append(epoch)
        with torch.inference_mode():
            test_predictions = model(X_test)
            test_loss = loss_function(test_predictions, Y_test)
            test_losses.append(test_loss.item())

model.eval()
print(model.state_dict())
with torch.inference_mode():
    pred = model(Y_test)
    
plot_data(predictions=pred.cpu())
plt.figure()
plt.plot(epochs, losses, label="training loss")
plt.plot(test_epochs, test_losses, label="test loss")
plt.legend()
plt.grid()
plt.show()

torch.save(model.state_dict(), "model_2.pth")
state = torch.load("model_2.pth")
model2 = LinearRegressionModel()
model2.to(device)
model2.load_state_dict(state)
print(model2.state_dict())


    