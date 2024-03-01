import torch
from torch import nn
import matplotlib.pyplot as plt

class LinearRegressionModel(nn.Module): # inherit from nn.Module -> almost everything in PyTorch inherits from nn.Module
    def __init__(self) -> None:
        super().__init__()

        # initialize model parameters
        self.weights = nn.Parameter(
            torch.randn(
                1,
                requires_grad=True, # PyTorch will track the gradients for use with torch.autograd
                dtype=torch.float32)
            )
        self.bias = nn.Parameter(
            torch.randn(
                1,
                requires_grad=True,
                dtype=torch.float32)
            )

    # used to compute output of the model (should be overridden in subclasses of nn.Module)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weights + self.bias # this is the linear regression equation
    

# load model from file
state = torch.load("model_0.pth")
model = LinearRegressionModel()
model.load_state_dict(state)
print(model.state_dict())
model.eval()
with torch.inference_mode():
    loaded_model_output = model(torch.tensor(0.0))
    print(loaded_model_output)