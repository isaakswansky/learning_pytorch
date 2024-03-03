import torch
import matplotlib.pyplot as plt

def tanh(x: torch.Tensor):
    return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

tensor = torch.arange(-10, 10, 0.1)
print(tensor)

t = tanh(tensor)
print(t)

plt.plot(t)
plt.show()