import torch
import torch.nn as nn
from einops import rearrange

from src.model.basic_block import ConvBnPReLu2d, GeMPool2d


class SimpleHeadV2(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        drop_rate: float = 0.0,
        kernel_size: int = 1,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBnPReLu2d(in_channels, hidden_channels, kernel_size=1),
            ConvBnPReLu2d(
                hidden_channels,
                hidden_channels,
                kernel_size=kernel_size,
                groups=hidden_channels,
            ),
            nn.Conv2d(hidden_channels, 6, kernel_size=1),
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
