import torch.nn as nn
from torch import Tensor


class PickLast(nn.Module):
    def __init__(
        self,
        encoder_channels: list[int],  # d0-d5
    ):
        super().__init__()

        self._output_size = encoder_channels[-1]

    @property
    def output_size(self) -> int:
        return self._output_size

    def forward(self, features: list[Tensor]) -> Tensor:
        return features[-1]
