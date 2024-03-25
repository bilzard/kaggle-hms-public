import torch.nn as nn
from torch import Tensor


class EegTimeCroppingTransform(nn.Module):
    def __init__(
        self,
        start: int,
        size: int,
    ):
        super().__init__()
        self.start = start
        self.size = size

    def forward(self, eeg: Tensor, eeg_mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        spec: b c t
        mask: b c t
        """
        eeg = eeg[..., self.start : self.start + self.size]
        eeg_mask = eeg_mask[..., self.start : self.start + self.size]
        return eeg, eeg_mask
