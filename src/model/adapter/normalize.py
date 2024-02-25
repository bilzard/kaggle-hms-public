import torch.nn as nn
from torch import Tensor

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


def mean_std_normalizer(x: Tensor, dim: tuple[int, ...], eps: float = 1e-4) -> Tensor:
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True)
    return (x - mean) / (std + eps)


class InstanceNormalizer(nn.Module):
    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, spec: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        spec: (B, C, F, T)
        """

        return mean_std_normalizer(spec, (2, 3), self.eps), mask


class LayerNormalizer(nn.Module):
    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, spec: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        spec: (B, C, F, T)
        """

        return mean_std_normalizer(spec, (1, 2, 3), self.eps), mask


class BatchNormalizer(nn.Module):
    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, spec: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        spec: (B, C, F, T)
        """

        return mean_std_normalizer(spec, (0, 2, 3), self.eps), mask
