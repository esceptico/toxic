import torch
from dataclasses import dataclass
from omegaconf import DictConfig
from torch import nn

from src.toxic.modelling.encoder import WideCNNEncoder
from src.toxic.modelling.classifier import SentenceClassifier


@dataclass
class ModelResult:
    embeddings: torch.Tensor
    logits: torch.Tensor


class Model(nn.Module):
    """A module that contains an encoder and a classification head"""
    def __init__(self, config: DictConfig):
        super().__init__()
        self.encoder = WideCNNEncoder(**config.model.encoder)
        self.classifier = SentenceClassifier(
            **config.model.classifier,
            n_classes=len(config.data.labels)
        )

    def forward(self, inputs) -> ModelResult:
        embeddings = self.encoder(inputs)
        logits = self.classifier(embeddings)
        return ModelResult(embeddings, logits)
