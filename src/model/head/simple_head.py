import torch
import torch.nn as nn
from einops import rearrange

from src.model.model_util import GeMPool2d


class SimpleHead(nn.Module):
    def __init__(self, in_channels: int, drop_rate: float = 0.0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, 6, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.pool = GeMPool2d()
        self.drop_rate = drop_rate
        self.dropout = nn.Dropout2d(drop_rate, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, in_channels, h, w)
        return: (batch_size, out_channels)
        """
        if self.drop_rate > 0:
            x = self.dropout(x)
        x = self.conv(x)
        x = self.pool(x)
        x = rearrange(x, "b c 1 1 -> b c")

        return x
