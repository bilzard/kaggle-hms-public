import numpy as np
import torch
import torch.nn as nn

CHANNEL_MEAN = [
    -37.27,
    -37.33,
    -37.4,
    -38.24,
    -37.18,
    -36.79,
    -37.9,
    -38.24,
    -37.28,
    -37.82,
    -36.86,
    -36.69,
    -37.86,
    -38.34,
    -37.16,
    -37.34,
    -37.33,
    -38.37,
]

CHANNEL_STD = [
    16.29,
    15.88,
    15.95,
    15.92,
    16.27,
    16.3,
    15.9,
    15.95,
    16.75,
    16.29,
    16.31,
    16.22,
    15.92,
    15.91,
    16.35,
    15.9,
    15.91,
    15.78,
]


class ConstantNormalizer(nn.Module):
    """
    固定の平均値と標準偏差でダイナミックレンジをスケールする
    """

    def __init__(
        self,
        mean: list[float] = CHANNEL_MEAN,
        std: list[float] = CHANNEL_STD,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.mean = torch.from_numpy(
            np.array(mean)[np.newaxis, :, np.newaxis, np.newaxis]
        ).to(dtype)
        self.std = torch.from_numpy(
            np.array(std)[np.newaxis, :, np.newaxis, np.newaxis]
        ).to(dtype)

    def forward(
        self, spec: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert spec.shape[1] == 18
        device = spec.device
        spec = (spec - self.mean.to(device)) / self.std.to(device)

        return spec, mask


def mean_std_normalizer(
    x: torch.Tensor, dim: tuple[int, ...], eps: float = 1e-4
) -> torch.Tensor:
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True)
    return (x - mean) / (std + eps)


class InstanceNormalizer(nn.Module):
    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def forward(
        self, spec: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        spec: (B, C, F, T)
        """

        return mean_std_normalizer(spec, (2, 3), self.eps), mask


class LayerNormalizer(nn.Module):
    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def forward(
        self, spec: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        spec: (B, C, F, T)
        """

        return mean_std_normalizer(spec, (1, 2, 3), self.eps), mask


class BatchNormalizer(nn.Module):
    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def forward(
        self, spec: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        spec: (B, C, F, T)
        """

        return mean_std_normalizer(spec, (0, 2, 3), self.eps), mask
