import torch.nn as nn
from einops import rearrange
from torch import Tensor

from src.model.basic_block import GatedMilAttention, GeMPool2d


class AttentionPoolingOverSamples(nn.Module):
    def __init__(self, input_channels: int, hidden_size: int = 64):
        super().__init__()
        self.pool = GeMPool2d()
        self.channel_mapper = nn.Conv2d(input_channels, hidden_size, kernel_size=1)
        self.att = GatedMilAttention(hidden_size=hidden_size)

        self._output_size = hidden_size

    @property
    def output_size(self) -> int:
        return self._output_size

    def forward(self, x: Tensor, num_samples: int) -> Tensor:
        """
        Attention Pooling over training/validation samples

        x: (B * S, C, T, F)

        return
        ------
        x: (B, C, 1, 1)
        """
        x = self.channel_mapper(x)
        x = self.pool(x)

        if num_samples == 1:
            return x

        x = x.squeeze(-1).squeeze(-1)
        x = rearrange(x, "(b s) c -> b c s", s=num_samples)
        x = (x * self.att(x)).sum(dim=-1)
        x = x.unsqueeze(-1).unsqueeze(-1)

        return x
