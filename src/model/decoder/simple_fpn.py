import torch
import torch.nn as nn
from torch import Tensor


class SimpleFpn(nn.Module):
    def __init__(
        self,
        encoder_channels: list[int],  # d0-d5
        kernel_size: int = 3,
        hidden_dim: int = 64,
        depth: int = 3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.channels = encoder_channels[-depth:][::-1]

        self.mappers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        hidden_dim,
                        kernel_size=kernel_size,
                        padding="same",
                    ),
                    nn.BatchNorm2d(hidden_dim),
                    nn.PReLU(),
                    nn.AvgPool2d(kernel_size=2**i, stride=2**i),
                )
                for i, in_channels in enumerate(self.channels)
            ]
        )
        self._output_size = self.hidden_dim * self.depth

    @property
    def output_size(self) -> int:
        return self._output_size

    def forward(self, features: list[Tensor]) -> Tensor:
        features = features[-self.depth :][::-1]
        for i, (mapper, feature) in enumerate(zip(self.mappers, features)):
            features[i] = mapper(feature)
        features = features[::-1]
        x = torch.cat(features, dim=1)
        return x
