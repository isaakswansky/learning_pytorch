import torch
import torch.nn as nn
import torchmetrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from helpers import plot_decision_boundary, train_and_test

NUM_FEATURES = 2
NUM_CLASSES = 3
RANDOM_SEED = 42
HIDDEN_LAYERS = 1
HIDDEN_LAYER_SIZE = 32
EPOCHS = 500
LEARNING_RATE = 0.01

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


def make_spirals(samples_per_class=100, dimensions=NUM_FEATURES, num_classes=NUM_CLASSES):
    N = samples_per_class # number of points per class
    D = dimensions # dimensionality
    K = num_classes # number of classes
    X = np.zeros((N*K,D)) # data matrix (each row = single example)
    y = np.zeros(N*K, dtype='uint8') # class labels
    for j in range(K):
        ix = range(N*j,N*(j+1))
        r = np.linspace(0.0,1,N) # radius
        t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
    return X, y

X_data, y_data = make_spirals()
print(y_data[:10])
# lets visualize the data
# plt.scatter(X_data[:, 0], X_data[:, 1], c=y_data, s=40, cmap=plt.cm.Spectral)
# plt.show()


X_data = torch.from_numpy(X_data).float().to(device)
y_data = torch.from_numpy(y_data).to(device)

# 2. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_data,
                                                    y_data,
                                                    test_size=0.2,
                                                    random_state=RANDOM_SEED)
print(y_train[:5])
# 3. Building a model
class SpiralModel(nn.Module):
    def __init__(self,
                 in_features=NUM_FEATURES,
                 out_features=NUM_CLASSES,
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

model = SpiralModel(
    activation=nn.ReLU()
    ).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(device)
metrics = {"accuracy": accuracy}

def to_prediction_labels(logits):
    probs = torch.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)
    return preds

train_and_test(
    epochs=EPOCHS,
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
