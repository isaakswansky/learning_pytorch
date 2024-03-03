
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np

def plot_decision_boundary(model: torch.nn.Module, X: torch.tensor, y: torch.tensor):
    """
    Plots the decision boundary of a model along with the data points.
    @param model: The model to use for prediction
    @param X: The data points
    @param y: The labels
    """
    # Copy data to CPU
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Calculate min and max values for X and y
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = torch.meshgrid(torch.linspace(x_min, x_max, 101),
                            torch.linspace(y_min, y_max, 101))
    
    # Create data to predict
    X_to_predict = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_predict)

    # Convert logits to predictions
    if len(torch.unique(y)) > 2:
        # If there are more than 2 classes, use argmax
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))

    y_pred = y_pred.reshape(xx.shape).detach().numpy()

    # Plot decision boundary
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())



def plot_predictions(train_data,
                     train_labels,
                     test_data,
                     test_labels,
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


def accuracy(y_train, y_pred):
    correct = torch.eq(y_train, y_pred).sum().item()
    acc = (correct / len(y_train)) * 100
    return acc


def do_nothing(x):
    return x

def train_and_test(
        epochs: int,
        model: nn.Module,
        loss_fn,
        optimizer,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        transformation=do_nothing,
        log_interval: int=10,
        metrics:dict=dict()):
    for epoch in range(epochs):
        # Activate training mode
        model.train()

        # Forward pass.squeeze()
        y_pred_raw = model(X_train).squeeze()

        # Calculate loss
        loss = loss_fn(y_pred_raw, y_train.squeeze()) # BCEWithLogitsLoss takes raw logits as input, not probabilities
        
        y_pred = transformation(y_pred_raw)

        # Calculate accuracy
        training_metrics = {}
        for metric_name, metric_fn in metrics.items():
            training_metrics[metric_name] = metric_fn(y_train, y_pred)

        # Zero the gradients
        optimizer.zero_grad()

        # Backward pass (backpropagation)
        loss.backward()

        # Update weights (gradient descent)
        optimizer.step()

        # Testing the model
        model.eval()
        with torch.inference_mode():
            y_test_raw = model(X_test).squeeze()
            y_test_pred = transformation(y_test_raw)
            test_loss = loss_fn(y_test_raw, y_test.squeeze())
            test_metrics = {}
            for metric_name, metric_fn in metrics.items():
                test_metrics[metric_name] = metric_fn(y_test, y_test_pred)

        # Print some results
        if (epoch) % log_interval == 0:
            print(f"====================== Epoch [{epoch + 1}/{epochs}] =========================")
            print(f" - Training:")
            print(f"     - Loss: {loss.item():.4f}")
            for metric_name, metric_value in training_metrics.items():
                print(f"     - {metric_name}: {metric_value:.4f}")
            print(f" - Testing:")
            print(f"     - Loss: {test_loss.item():.4f}")
            for metric_name, metric_value in test_metrics.items():
                print(f"     - {metric_name}: {metric_value:.4f}")