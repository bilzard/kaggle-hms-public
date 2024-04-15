import torch.nn as nn
from einops import rearrange
from torch import Tensor


class MeanAggregator(nn.Module):
    def __init__(self, input_channels: int):
        super().__init__()
        self._output_size = input_channels

    @property
    def output_size(self) -> int:
        return self._output_size

    def forward(self, x: Tensor, num_samples: int) -> Tensor:
        x = rearrange(x, "(b s) c f t -> b s c f t", s=num_samples)
        if num_samples == 1:
            return x[:, 0, ...]

        x = x.mean(dim=1)
        return x
