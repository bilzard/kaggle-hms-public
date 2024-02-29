import torch.nn as nn
from torch import Tensor

from src.model.basic_block import norm_mean_std

CHANNEL_MEAN = -37.52
CHANNEL_STD = 16.10


class ConstantNormalizer(nn.Module):
    """
    固定の平均値と標準偏差でダイナミックレンジをスケールする
    """

    def __init__(
        self,
        mean: float = CHANNEL_MEAN,
        std: float = CHANNEL_STD,
    ):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, spec: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        spec = (spec - self.mean) / self.std

        return spec, mask


class InstanceNormalizer(nn.Module):
    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, spec: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        spec: (B, C, F, T)
        """

        return norm_mean_std(spec, (2, 3), self.eps), mask


class LayerNormalizer(nn.Module):
    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, spec: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        spec: (B, C, F, T)
        """

        return norm_mean_std(spec, (1, 2, 3), self.eps), mask


class BatchNormalizer(nn.Module):
    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, spec: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        spec: (B, C, F, T)
        """

        return norm_mean_std(spec, (0, 2, 3), self.eps), mask
