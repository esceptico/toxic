from typing import Sequence, Tuple

import torch
from torch import nn

from src.toxic.modelling.modules import Conv1dMaxPooling


class WideCNNEncoder(nn.Module):
    """Convolutional sentence encoder

    References:
        Convolutional Neural Networks for Sentence Classification
        https://arxiv.org/abs/1408.5882
    """
    def __init__(
        self,
        token_embedding_size: int = 32,
        vocab_size: int = 256,
        filters: Sequence[Tuple[int, int]] = ((1, 4), (2, 8), (3, 16)),
        dropout: float = 0.2,
        projection_size: int = 256
    ):
        """Constructor

        Args:
            token_embedding_size (int): Size of token embedding.
                Defaults to `32`.
            vocab_size (int): Number of token dictionary. Defaults to `256`.
            filters (Sequence[Tuple[int, int]]): Sequence of
                [kernel_size, out_channels] tuples.
            dropout (float): Dropout value.
            projection_size (int): Output layer size. Defaults to `256`
        """
        super().__init__()
        self.projection_size = projection_size
        self.token_embedding = nn.Embedding(
            vocab_size, token_embedding_size, padding_idx=0
        )
        self.convolutions = nn.ModuleList([
            Conv1dMaxPooling(
                in_channels=token_embedding_size,
                out_channels=out_size,
                kernel_size=width
            ) for width, out_size in filters
        ])
        projection_input_size = sum(out_size for _, out_size in filters)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(projection_input_size, projection_size)

    def forward(self, inputs):
        token_embedding = self.token_embedding(inputs).transpose(1, 2)
        conv = [conv(token_embedding) for conv in self.convolutions]
        conv = torch.cat(conv, dim=-1)
        conv = self.dropout(conv)
        return self.projection(conv)
