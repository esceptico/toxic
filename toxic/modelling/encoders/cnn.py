from typing import Sequence, Tuple

import torch
from torch import nn

from toxic.modelling.modules import Conv1dMaxPooling
from toxic.modelling.encoders.base import SentenceEncoder


class WideCNNEncoder(SentenceEncoder):
    def __init__(
        self,
        token_embedding_size: int = 32,
        vocab_size: int = 256,
        filters: Sequence[Tuple[int, int]] = ((1, 4), (2, 8), (3, 16)),
        dropout: float = 0.2,
        projection_size: int = 256
    ):
        super().__init__(embedding_size=projection_size)
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
