import torch
import matplotlib.pyplot as plt

A = torch.arange(-10, 10, 1, dtype=torch.float32)

def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.tensor(0, dtype=torch.float32), x)

def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-x))

sig = sigmoid(A)
rel = relu(A)

# normalize
sig /= sig.max()
rel /= rel.max()

plt.plot(sig, label="sigmoid")
plt.plot(rel, label="relu")
plt.legend()
plt.grid()
plt.show()