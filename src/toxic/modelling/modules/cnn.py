import torch
from torch import nn


class Conv1dMaxPooling(nn.Module):
    """Conv1d -> MaxPooling1d -> Activation"""
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        activation=torch.relu
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size
        )
        self.activation = activation

    def forward(self, inputs):
        out = self.conv(inputs)
        out, _ = torch.max(out, dim=-1)
        return self.activation(out)
