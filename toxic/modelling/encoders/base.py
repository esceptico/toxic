from torch import nn


class SentenceEncoder(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.embedding_size = embedding_size

    def requires_grad(self, condition: bool) -> 'SentenceEncoder':
        for p in self.parameters():
            p.requires_grad = condition
        return self
