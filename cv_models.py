import torch.nn as nn

class FashionMNISTCNNV0(nn.Module):
    """
    Model architecture that replicates the TinyVGG model from CNN explainer website.
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7,
                      out_features=output_shape),
        )
    
    def forward(self, x):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))



class BasicFashionMNISTModel(nn.Module):
    def __init__(
            self,
            in_features: int = 0,
            out_features: int = 0,
            hidden_layers: int = 0, 
            hidden_layer_size: int = 0,
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

class FashionMNISTModelV0(BasicFashionMNISTModel):
    def __init__(self):
        super().__init__(
            in_features=28*28,
            out_features=10,
            hidden_layers=1,
            hidden_layer_size=10,
        )

class FashionMNISTModelV1(BasicFashionMNISTModel):
    def __init__(self):
        super().__init__(
            in_features=28*28,
            out_features=10,
            hidden_layers=1,
            hidden_layer_size=10,
            activation=nn.ReLU()
        )