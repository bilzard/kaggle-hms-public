import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor


def dual_stack_eeg_channels(
    x: Tensor, drop_z: bool = False, plus_one: bool = False
) -> Tensor:
    """
    spec: (b, 18, t)
    mask: (b, 18, t)

    Return:
    spec: (2, b, 10, t)
    mask: (2, b, 10, t)
    """
    ll = x[:, 0:4]
    lp = x[:, 4:8]
    z = x[:, 8:10]
    rp = x[:, 10:14]
    rl = x[:, 14:18]

    left = torch.cat([ll, lp], dim=1)
    right = torch.cat([rl, rp], dim=1)

    if not drop_z:
        left = torch.cat([left, z], dim=1)
        right = torch.cat([right, z], dim=1)

    if plus_one:
        z_comp = -(z[:, 0] + z[:, 1]).unsqueeze(1)  # Pz-Fz
        left = torch.cat([left, z_comp], dim=1)
        right = torch.cat([right, z_comp], dim=1)

    x = torch.stack([left, right], dim=0)

    return x


class EegDualStackingCollator(nn.Module):
    def __init__(self, drop_z: bool = False, plus_one: bool = False):
        super().__init__()
        self.drop_z = drop_z
        self.plus_one = plus_one

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(drop_z={self.drop_z}, plus_one={self.plus_one})"
        )

    def forward(self, eeg: Tensor, eeg_mask: Tensor) -> tuple[Tensor, Tensor]:
        eeg = dual_stack_eeg_channels(eeg, drop_z=self.drop_z, plus_one=self.plus_one)
        eeg_mask = dual_stack_eeg_channels(
            eeg_mask, drop_z=self.drop_z, plus_one=self.plus_one
        )
        eeg = rearrange(eeg, "d b c t -> (d b) c t")
        eeg_mask = rearrange(eeg_mask, "d b c t -> (d b) c t")

        return eeg, eeg_mask
