from torch import nn

from toxic.modelling.encoders import SentenceEncoder


class ShallowSentenceClassifier(nn.Module):
    def __init__(
        self,
        encoder: SentenceEncoder,
        dropout: float = 0.5,
        projection_size: int = 256,
        n_classes: int = 4
    ):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.encoder.embedding_size, projection_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_size, n_classes)
        )

    def forward(self, inputs):
        embedding = self.encoder(inputs)
        return self.head(embedding)
