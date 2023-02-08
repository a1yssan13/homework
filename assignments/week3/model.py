import torch
import torch.nn as nn
from typing import Callable


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super(MLP, self).__init__()
        self.act = activation()
        self.first = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(hidden_count)]
        )
        self.last = nn.Linear(hidden_size, num_classes)
        initializer(self.first.weight)
        for layer in self.hidden_layers:
            initializer(layer.weight)
        initializer(self.last.weight)

    def forward(self, x):
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        x = self.act(self.first(x))
        for layer in self.hidden_layers:
            x = self.act(layer(x))
        x = self.last(x)
        return x
