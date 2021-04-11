from typing import Callable

import torch
from torch import nn


class Highway(nn.Module):
    """
    Highway Networks
    https://arxiv.org/abs/1505.00387

    A highway network does a gated combination of a linear
    transformation and a non-linear transformation of its input.
    """
    def __init__(
        self,
        input_dim: int,
        num_layers: int = 1,
        activation: Callable[[torch.Tensor], torch.Tensor] = torch.relu
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.activation = activation

        self.nonlinear = nn.ModuleList([
            nn.Linear(input_dim, input_dim) for _ in range(num_layers)
        ])
        self.gate = nn.ModuleList([
            nn.Linear(input_dim, input_dim) for _ in range(num_layers)
        ])
        for layer in self.gate:
            # Gate bias can be initialized with a negative
            # value (e.g. -1, -3 etc.) such that the network is initially
            # biased towards carry behavior
            layer.bias.data.fill_(-1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        current_input = inputs
        for layer in range(self.num_layers):
            linear = current_input
            nonlinear = self.nonlinear[layer](current_input)
            nonlinear = self.activation(nonlinear)
            gate = self.gate[layer](current_input)
            gate = torch.sigmoid(gate)
            current_input = gate * nonlinear + (1 - gate) * linear
        return current_input
