from torch import nn


class SentenceClassifier(nn.Module):
    """Classification head"""

    def __init__(
        self,
        embedding_size: int,
        dropout: float = 0.5,
        hidden_size: int = 256,
        n_classes: int = 4
    ):
        """Constructor

        Args:
            embedding_size (int): Size of encoder embedding.
            dropout (float): Dropout probability.
            hidden_size (int): Size of hidden layer.
            n_classes (int): Output size (i.e. number of classes).
        """
        super().__init__()
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, inputs):
        return self.head(inputs)
