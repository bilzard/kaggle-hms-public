import torch.nn as nn
from torch import Tensor


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        spec: Tensor,
        spec_mask: Tensor,
        eeg: Tensor,
        eeg_mask: Tensor,
        label: Tensor,
        weight: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        return (spec, spec_mask, eeg, eeg_mask, label, weight)
