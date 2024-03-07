import torch
import torch.nn as nn


class Head(nn.Module):
    def __init__(self, in_channels: int, bottleneck_ratio: int = 4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // bottleneck_ratio),
            nn.BatchNorm1d(in_channels // bottleneck_ratio),
            nn.PReLU(),
            nn.Linear(in_channels // bottleneck_ratio, 6, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: b c
        return: b 6
        """
        return self.mlp(x)
