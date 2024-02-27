import torch
import torch.nn as nn
from torch import Tensor


class ClampedTanh(nn.Module):
    def __init__(self, delta: float = 1e-2):
        super().__init__()
        self.delta = delta

    def forward(self, x: Tensor) -> Tensor:
        return torch.clamp(x.tanh(), -1 + self.delta, 1 - self.delta)


class ClampedSigmoid(nn.Module):
    def __init__(self, delta: float = 1e-2):
        super().__init__()
        self.delta = delta

    def forward(self, x: Tensor) -> Tensor:
        return torch.clamp(x.sigmoid(), self.delta, 1 - self.delta)
