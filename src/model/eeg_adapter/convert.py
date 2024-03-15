import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


def mu_law_encoding(x: Tensor, mu: float) -> Tensor:
    return torch.sign(x) * torch.log1p(mu * torch.abs(x)) / np.log1p(mu)


class MuLawEncoding(nn.Module):
    def __init__(self, mu: float, T: float):
        super().__init__()
        self.T = T
        self.mu = mu

    def forward(self, eeg: Tensor, eeg_mask: Tensor) -> tuple[Tensor, Tensor]:
        eeg = mu_law_encoding(eeg / self.T, self.mu)
        return eeg, eeg_mask
