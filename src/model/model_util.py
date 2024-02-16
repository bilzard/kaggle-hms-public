import torch
import torch.nn as nn


class InverseSoftmax(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-4, C: float = 0.0):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.C = nn.Parameter(torch.ones(1) * C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x + self.eps) * self.C
