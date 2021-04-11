from torch import nn


class MeanPooling(nn.Module):
    def forward(self, inputs, mask=None):
        if mask is None:
            return inputs.mean(1)
        masked_inputs = inputs * mask.unsqueeze(-1)
        pooled = masked_inputs.sum(1) / mask.sum(-1).unsqueeze(-1)
        return pooled
