import torch
import torch.nn as nn
import torchmetrics
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from tqdm.auto import tqdm

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from cv_models import FashionMNISTModelV0, FashionMNISTModelV1

RANDOM_SEED = 42
BATCH_SIZE = 32

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None
)
test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False
    )

device = "cuda" if torch.cuda.is_available() else "cpu"

model = FashionMNISTModelV0().to(device)
loaded_state = torch.load("cv_models/" + model.__class__.__name__ + ".pth")
model.load_state_dict(loaded_state)

loss_fn = nn.CrossEntropyLoss() # multiclass classification -> CLE
metric = torchmetrics.Accuracy(task="multiclass", num_classes=len(test_data.classes)).to(device)

# evaluate model
torch.manual_seed(RANDOM_SEED)
def eval_model(
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        accuracy_fn,
        device=device
        ):
    loss = 0
    accuracy = 0
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            accuracy += accuracy_fn(y, y_pred.argmax(dim=1))
        # Scale the loss and accuracy by the number of batches
        loss /= len(data_loader)
        accuracy /= len(data_loader)
    
    return {
        "model_name": model.__class__.__name__,
        "model_loss": loss.item(),
        "model_accuracy": accuracy.item()
    }

# Evaluate the model
model_0_eval = eval_model(
    model=model, 
    data_loader=test_loader,
    loss_fn=loss_fn,
    accuracy_fn=metric)
print(model_0_eval)