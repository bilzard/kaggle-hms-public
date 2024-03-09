import torch
import torch.nn as nn
import torch.nn.functional as F


class GeMPool1d(nn.Module):
    def __init__(self, p: int = 3, eps: float = 1e-4):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: b c t
        """
        return F.adaptive_avg_pool1d(x.clamp(min=self.eps).pow(self.p), 1).pow(
            1.0 / self.p
        )


class GeMPool2d(nn.Module):
    def __init__(self, p: int = 3, eps: float = 1e-4):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: b c f t
        """
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), (1, 1)).pow(
            1.0 / self.p
        )


class GeMPool3d(nn.Module):
    def __init__(self, p: int = 3, eps: float = 1e-4):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: b c ch f t
        """
        return F.adaptive_avg_pool3d(x.clamp(min=self.eps).pow(self.p), (1, 1, 1)).pow(
            1.0 / self.p
        )
