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
from cv_models import FashionMNISTModelV0, FashionMNISTModelV1, FashionMNISTCNNV0
from helpers import cv_train_step, cv_test_step

# 1. Getting dataset
# we will use the FashionMNIST dataset from torchvision
train_data = datasets.FashionMNIST(
    root="data", # where to download data to?
    train=True, # is this the training set?
    download=True, # should we download the data?
    transform=ToTensor(), # how to transform the data?
    target_transform=None # how to transform the target?
)
test_data = datasets.FashionMNIST(
    root="data", # where to download data to?
    train=False, # is this the training set?
    download=True, # should we download the data?
    transform=ToTensor(), # how to transform the data?
    target_transform=None # how to transform the target?
)
print("Classes: ", train_data.classes)

# Exploring the dataset
print(train_data.data[0].shape) # shape of the first image
image, label = train_data[0]
print(f"Image shape: {image.shape}")

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_data), size=(1,)).item()
    img, label = train_data[sample_idx]
    # fig.add_subplot(rows, cols, i)
    # plt.title(train_data.classes[label])
    # plt.axis("off")
    # plt.imshow(img.squeeze(), cmap="gray")

# plt.show()

# 2. Prepare data loader
# DataLoader combines a dataset and a sampler, and provides an iterable over the given dataset
# We want to split the data into batches
# Why would we do this?
# - We can't fit all the data into memory at once
# - We can update the model weights after each batch (not only after each epoch)

BATCH_SIZE = 32
train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
    )
test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False
    )
print(f"Number of training batches: {len(train_loader)}")

# Look at a random image from the batch
train_features_batch, train_labels_batch = next(iter(train_loader))
print(f"Feature batch shape: {train_features_batch.size()}")
random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
img, label = train_features_batch[random_idx], train_labels_batch[random_idx]

# plt.imshow(img.squeeze(), cmap="gray")
# plt.title(train_data.classes[label.item()])
# plt.axis("off")
# plt.show()
    
model_0 = FashionMNISTCNNV0(
    input_shape=1,
    hidden_units=10,
    output_shape=len(train_data.classes)
).to(device)

print(model_0.state_dict())
loss_fn = nn.CrossEntropyLoss() # multiclass classification -> CLE
optimizer = torch.optim.SGD(model_0.parameters(), lr=0.1)
metric = torchmetrics.Accuracy(task="multiclass", num_classes=len(train_data.classes)).to(device)

def print_train_time(start:float, end:float, device:str):
    print(f"Training time on {device}: {end - start:.2f} seconds")

# Create a training loop
train_time_start = timer()
epochs = 3

for epoch in range(epochs):
    print(f"Epoch {epoch+1} ======================")
    cv_train_step(
        model=model_0,
        data_loader=train_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=metric,
        device=device
    )
    cv_test_step(
        model=model_0,
        data_loader=test_loader,
        loss_fn=loss_fn,
        accuracy_fn=metric,
        device=device
    )

train_time_end = timer()
print_train_time(train_time_start, train_time_end, device)

# Save the model
torch.save(model_0.state_dict(), "cv_models/" + model_0.__class__.__name__ + ".pth")