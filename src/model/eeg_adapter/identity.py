import torch.nn as nn
from torch import Tensor


class Identity(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, eeg: Tensor, eeg_mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        spec: b c t
        mask: b c t
        """
        return eeg, eeg_mask
