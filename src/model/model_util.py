import torch
import torch.nn as nn
import torch.nn.functional as F


class GeMPool2d(nn.Module):
    def __init__(self, p: int = 3, eps: float = 1e-4):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        _, _, h, w = x.shape
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), (1, 1)).pow(
            1.0 / self.p
        )


class InverseSoftmax(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-4, C: float = 0.0):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.C = nn.Parameter(torch.ones(1) * C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x + self.eps) * self.C
