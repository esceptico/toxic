import torch
from torch import nn
from captum.attr import LayerIntegratedGradients


def reduce_embedding_attributions(attributions: torch.Tensor) -> torch.Tensor:
    outputs = attributions.sum(dim=2).squeeze(0)
    outputs = outputs / torch.norm(outputs)
    outputs = outputs.cpu().detach()
    return outputs


def lig_explain(
    inputs: torch.Tensor,
    target,
    model: nn.Module,
    embedding_layer: nn.Module
) -> torch.Tensor:
    lig = LayerIntegratedGradients(model.forward, embedding_layer)
    attributions = lig.attribute(inputs, target=target)
    attributions = reduce_embedding_attributions(attributions)
    return attributions
