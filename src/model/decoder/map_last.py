import torch.nn as nn
from torch import Tensor


class MapLast(nn.Module):
    def __init__(
        self,
        encoder_channels: list[int],  # d0-d5
        hidden_dim: int,
        activation: nn.Module,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.mapper = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            activation,
        )

    @property
    def output_size(self) -> int:
        return self.hidden_dim

    def forward(self, features: list[Tensor]) -> Tensor:
        return self.mapper(features[-1])
