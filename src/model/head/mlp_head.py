import torch
import torch.nn as nn
from einops import rearrange


class Head(nn.Module):
    def __init__(
        self,
        in_channels: int,
        bottleneck_ratio: int = 4,
        num_heads: int = 2,
        num_classes: int = 6,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // bottleneck_ratio),
            nn.BatchNorm1d(in_channels // bottleneck_ratio),
            nn.PReLU(),
            nn.Linear(
                in_channels // bottleneck_ratio, num_heads * num_classes, bias=True
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: b c
        return: b k 6
        """
        x = self.mlp(x)
        x = rearrange(x, "b (k c) -> b k c", k=self.num_heads)
        return x
