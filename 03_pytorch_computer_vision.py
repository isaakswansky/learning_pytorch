import torch
import torch.nn as nn
import torchmetrics
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from tqdm.auto import tqdm

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
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

# Exploring the dataset
print(train_data.data[0].shape) # shape of the first image
image, label = train_data[0]
print(f"Image shape: {image.shape}")

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"

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
print(f"Number of test batches: {len(test_loader)}")

# Look at a random image from the batch
train_features_batch, train_labels_batch = next(iter(train_loader))
print(f"Feature batch shape: {train_features_batch.size()}")
random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
img, label = train_features_batch[random_idx], train_labels_batch[random_idx]

# plt.imshow(img.squeeze(), cmap="gray")
# plt.title(train_data.classes[label.item()])
# plt.axis("off")
# plt.show()

# 3. Building a baseline model (start simple and add complexity as needed)
class BasicFashionMNISTModel(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            hidden_layers: int, 
            hidden_layer_size: int,
            activation: nn.Module = None
            ):
        super().__init__()
        self.ff_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_features, out_features=hidden_layer_size)
        )
        if activation is not None:
            self.ff_layers.append(activation)
        
        for _ in range(hidden_layers):
            self.ff_layers.append(
                nn.Linear(in_features=hidden_layer_size, out_features=hidden_layer_size)
            )
            if activation is not None:
                self.ff_layers.append(activation)
        self.ff_layers.append(
            nn.Linear(in_features=hidden_layer_size, out_features=out_features)
        )

    def forward(self, x):
        return self.ff_layers(x)
    
model_0 = BasicFashionMNISTModel(
    in_features=28*28,
    out_features=len(train_data.classes),
    hidden_layers=1,
    hidden_layer_size=10,
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

for epoch in tqdm(range(epochs)):
    print(f"Epoch {epoch+1} ======================")

    train_loss = 0.0
    for batch, (X, y) in enumerate(train_loader):
        model_0.train()
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model_0(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # Zero gradients, backward pass, update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # Log every N batches
        if batch % 400 == 0:
            print(f"-------- batch {batch}------------")
            print(f"Looked at {batch * BATCH_SIZE} samples")

    train_loss /= len(train_loader)
    print(f"Training loss: {train_loss:.4f}")

    test_loss, test_acc = 0.0, 0.0
    model_0.eval()
    with torch.inference_mode():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            y_pred_test = model_0(X_test)
            loss = loss_fn(y_pred_test, y_test)
            test_loss += loss.item()
            test_acc += metric(y_test, y_pred_test.argmax(dim=1))
        test_loss /= len(test_loader)
        test_acc /= len(test_loader)

    print(f"Test loss: {test_loss:.4f}, test accuracy: {test_acc:.4f}")

train_time_end = timer()
print_train_time(train_time_start, train_time_end, device)