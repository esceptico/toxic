from typing import Any, Callable

import torch
from torch import nn
from captum.attr import LayerIntegratedGradients


def reduce_embedding_attributions(attributions: torch.Tensor) -> torch.Tensor:
    """Reduces attributions of embedding weights to token level

    Args:
        attributions (torch.Tensor): Embeddings layer attributions
            (from `LayerIntegratedGradients.attribute` method)

    Returns:
        torch.Tensor: Token-wise attributions
    """
    outputs = attributions.sum(dim=2).squeeze(0)
    outputs = outputs / torch.norm(outputs)
    outputs = outputs.cpu().detach()
    return outputs


def lig_explain(
    inputs: Any,
    target: int,
    forward: Callable,
    embedding_layer: nn.Module
) -> torch.Tensor:
    """Interpretability algorithm (Integrated Gradients) that assigns
    an importance score to each input token

    Args:
        inputs: Input for token embedding layer.
        target (int): Index of label for interpretation.
        forward (Callable): The forward function of the model or any
            modification of it.
        embedding_layer: Token embedding layer for which attributions are
            computed.

    Returns:
        Tensor of importance score to each input token
    """
    lig = LayerIntegratedGradients(forward, embedding_layer)
    attributions = lig.attribute(inputs, target=target)
    attributions = reduce_embedding_attributions(attributions)
    return attributions
