import torch
from torch import nn

from toxic.modelling.nn import Conv1dMaxPooling, Highway


class WideCNNEmbedder(nn.Module):
    def __init__(
        self,
        token_embedding_size=32,
        vocab_size=256,
        filters=((1, 4), (2, 8), (3, 16), (4, 32), (5, 64)),
        projection_size=256
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, token_embedding_size, padding_idx=0)
        self.convolutions = nn.ModuleList([
            Conv1dMaxPooling(
                in_channels=token_embedding_size,
                out_channels=out_size,
                kernel_size=width
            ) for width, out_size in filters
        ])
        # projection_input_size = sum(out_size for _, out_size in filters)
        # self.highway = Highway(projection_input_size)
        # self.projection = nn.Linear(projection_input_size, projection_size)
        self.output_size = sum(out_size for _, out_size in filters)

    def forward(self, inputs):
        token_embedding = self.token_embedding(inputs)     # (batch_size, seq_len, embedding_size)
        token_embedding = token_embedding.transpose(1, 2)  # (batch_size, embedding_size, seq_len)
        convolutions = [conv(token_embedding) for conv in self.convolutions]
        convolutions = torch.cat(convolutions, dim=-1)
        # highway = self.highway(convolutions)
        # return self.projection(highway)
        return convolutions


class Model(nn.Module):
    def __init__(self, token_emb_size=32, vocab_size=1024, filters=((1, 64), (2, 128)),
                 dropout=0.5, projection_size=256, n_classes=4):
        super().__init__()
        self.embedding = WideCNNEmbedder(
            token_embedding_size=token_emb_size,
            vocab_size=vocab_size,
            filters=filters,
            projection_size=projection_size
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.embedding.output_size, projection_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_size, n_classes)
        )

    def forward(self, inputs):
        x = self.embedding(inputs)
        return self.head(x)
