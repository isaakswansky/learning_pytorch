import torch
import torch.nn as nn
import torchmetrics
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from helpers import plot_decision_boundary, train_and_test

NUM_FEATURES = 2
NUM_SAMPLES = 1000
RANDOM_SEED = 42
HIDDEN_LAYERS = 1
HIDDEN_LAYER_SIZE = 10

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# 1. Create multiclass data
X_data, y_data = make_moons(n_samples=NUM_SAMPLES,
                            noise=0.2,
                            random_state=RANDOM_SEED
                            )
X_data = torch.from_numpy(X_data).float().to(device)
y_data = torch.from_numpy(y_data).float().to(device)

# plt.figure(figsize=(10, 7))
# plt.scatter(X_data[:, 0].cpu(), X_data[:, 1].cpu(), c=y_data.cpu(), cmap=plt.cm.RdYlBu)
# plt.show()

# 2. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_data,
                                                    y_data,
                                                    test_size=0.2,
                                                    random_state=RANDOM_SEED)
print(y_train[:5])
# 3. Building a model
class MoonModel(nn.Module):
    def __init__(self,
                 in_features=NUM_FEATURES,
                 out_features=1,
                 hidden_layers=HIDDEN_LAYERS,
                 hidden_layer_size=HIDDEN_LAYER_SIZE,
                 activation=None
                 ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_layer_size)
        )
        if activation is not None:
            self.layers.append(activation)
        
        for _ in range(hidden_layers):
            self.layers.append(
                nn.Linear(in_features=hidden_layer_size, out_features=hidden_layer_size)
            )
            if activation is not None:
                self.layers.append(activation)
        self.layers.append(
            nn.Linear(in_features=hidden_layer_size, out_features=out_features)
        )

    def forward(self, x):
        return self.layers(x)

model = MoonModel(
    activation=nn.ReLU()
    ).to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

accuracy = torchmetrics.Accuracy(task="binary").to(device)
metrics = {"accuracy": accuracy}

def to_prediction_labels(logits):
    probs = torch.sigmoid(logits)
    preds = torch.round(probs)
    return preds

train_and_test(
    epochs=1000,
    model=model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    loss_fn=loss_fn,
    optimizer=optimizer,
    transformation=to_prediction_labels,
    metrics=metrics,
    log_interval=10
)


fig = plt.figure(figsize=(12, 6))
fig.suptitle("Model 1")
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model, X_test, y_test)
plt.show()
