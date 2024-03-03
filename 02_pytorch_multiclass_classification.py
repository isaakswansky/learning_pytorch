import torch
from torch import nn
import matplotlib.pyplot as plt
import torchmetrics
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from helpers import plot_decision_boundary, train_and_test

# Set the hyperparameters for data creation
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42
HIDDEN_UNITS = 8
HIDDEN_LAYERS = 1
NUM_SAMPLES = 1000
LEARNING_RATE = 0.1

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

# 1. Create multiclass data
X_blob, y_blob = make_blobs(n_samples=NUM_SAMPLES,
                            n_features=NUM_FEATURES,
                            centers=NUM_CLASSES,
                            random_state=RANDOM_SEED,
                            cluster_std=1.5 # add some noise
                            )


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# 2. Turn data into tensors on the selected device
X_blob = torch.from_numpy(X_blob).float().to(device)
y_blob = torch.from_numpy(y_blob).long().to(device)

# 3. Split into training and test sets
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob,
                                                                        y_blob,
                                                                        test_size=0.2,
                                                                        random_state=RANDOM_SEED)

# 4. Plot the data
# plt.figure(figsize=(10, 7))
# plt.scatter(X_blob[:, 0].cpu(), X_blob[:, 1].cpu(), c=y_blob.cpu(), cmap=plt.cm.RdYlBu)
# plt.show()

# 5. Building a model
class BlobModel(nn.Module):
    def __init__(self,
                 input_features: int=NUM_FEATURES,
                 output_features: int=NUM_CLASSES,
                 hidden_units: int=HIDDEN_UNITS,
                 hidden_layers:int=HIDDEN_LAYERS,
                 activation=None):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units)
        )
        if activation is not None:
            self.linear_layer_stack.append(activation)

        for _ in range(hidden_layers):
            self.linear_layer_stack.append(nn.Linear(in_features=hidden_units, out_features=hidden_units))
            if activation is not None:
                self.linear_layer_stack.append(activation)

        self.linear_layer_stack.append(nn.Linear(in_features=hidden_units, out_features=output_features))

    def forward(self, x):
        return self.linear_layer_stack(x)

# 6. Train the model
model_0 = BlobModel(input_features=NUM_FEATURES,
                    output_features=NUM_CLASSES,
                    hidden_units=HIDDEN_UNITS,
                    hidden_layers=HIDDEN_LAYERS,
                    # activation=nn.ReLU()
                    ).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_0.parameters(), lr=LEARNING_RATE)

model_0.eval()
with torch.inference_mode():
    y_logits = model_0(X_blob_test)
print(y_logits[:10])
# in order to evaluate the model, we need to convert the logits to predictions

def to_prediction_labels(logits):
    probs = torch.softmax(logits, dim=1)
    return probs.argmax(dim=1)

print(to_prediction_labels(y_logits[:10]))

# Track some metrics
acc = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(device)
precision = torchmetrics.Precision(task="multiclass", num_classes=NUM_CLASSES).to(device)
recall = torchmetrics.Recall(task="multiclass", num_classes=NUM_CLASSES).to(device)

train_and_test(model=model_0,
               X_train=X_blob_train,
               y_train=y_blob_train,
               X_test=X_blob_test,
               y_test=y_blob_test,
               loss_fn=loss_fn,
               optimizer=optimizer,
               epochs=100,
               transformation=to_prediction_labels,
               metrics={"accuracy": acc, "precision": precision, "recall": recall},
               log_interval=10
               )


model_0.eval()
with torch.inference_mode():
    y_logits = model_0(X_blob_test)
    y_preds = to_prediction_labels(y_logits)
    print(y_preds[:10])
# Visualize model 1
fig = plt.figure(figsize=(12, 6))
fig.suptitle("Model 0")
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_0, X_blob_train, y_blob_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_0, X_blob_test, y_blob_test)
plt.show()

# A few more classification metrics...
# - Accuracy
# - Precision
# - Recall
# - F1 score
# - Confusion matrix
# - Classification report

print(acc(y_preds, y_blob_test))