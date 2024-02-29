import torch.nn as nn
from torch import Tensor

from src.model.basic_block import norm_mean_std

BG_CHANNEL_MEAN = -31.65
BG_CHANNEL_STD = 12.66


class BgConstantNormalizer(nn.Module):
    """
    固定の平均値と標準偏差でダイナミックレンジをスケールする
    """

    def __init__(
        self,
        mean: float = BG_CHANNEL_MEAN,
        std: float = BG_CHANNEL_STD,
    ):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, spec: Tensor) -> Tensor:
        spec = (spec - self.mean) / self.std

        return spec


class BgInstanceNormalizer(nn.Module):
    """
    F, Tの次元の統計を使ってmean-std normalizationする
    """

    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, spec: Tensor) -> Tensor:
        """
        spec: (B, C, F, T)
        """

        return norm_mean_std(spec, (2, 3), self.eps)


class BgLayerNormalizer(nn.Module):
    """
    C, F, Tの次元の統計を使ってmean-std normalizationする
    """

    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, spec: Tensor) -> Tensor:
        """
        spec: (B, C, F, T)
        """

        return norm_mean_std(spec, (1, 2, 3), self.eps)


class BgBatchNormalizer(nn.Module):
    """
    B, F, Tの次元の統計を使ってmean-std normalizationする
    """

    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, spec: Tensor) -> Tensor:
        """
        spec: (B, C, F, T)
        """

        return norm_mean_std(spec, (0, 2, 3), self.eps)
