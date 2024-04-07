import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor


def dual_partial_stack_eeg_channels(
    x: Tensor, xm: Tensor, drop_z: bool = False, eps: float = 1e-5
) -> tuple[Tensor, Tensor]:
    """
    channelグループとにまとめる

    spec: b ch t
    mask: b ch t

    Return:
    spec: (d b) ch g t
    mask: (d b) ch g t
    """
    ll = x[:, 0:4].unsqueeze(dim=2)  # b ch 1 t
    lp = x[:, 4:8].unsqueeze(dim=2)  # b ch 1 t
    z = x[:, 8:10].unsqueeze(dim=2)  # b ch 1 t
    rp = x[:, 10:14].unsqueeze(dim=2)  # b ch 1 t
    rl = x[:, 14:18].unsqueeze(dim=2)  # b ch 1 t

    llm = xm[:, 0:4].unsqueeze(dim=2)  # b ch 1 t
    lpm = xm[:, 4:8].unsqueeze(dim=2)  # b ch 1 t
    zm = xm[:, 8:10].unsqueeze(dim=2)  # b ch 1 t
    rpm = xm[:, 10:14].unsqueeze(dim=2)  # b ch 1 t
    rlm = xm[:, 14:18].unsqueeze(dim=2)  # b ch 1 t

    z_diff = (z[:, 0] - z[:, 1]).abs()
    z_wgt = z[:, 0] * zm[:, 0] + z[:, 1] * zm[:, 1] / (zm[:, 0] + zm[:, 1] + eps)

    z_feat = torch.stack([z_diff, z_wgt], dim=1)
    zz = torch.cat([z, z_feat], dim=1)  # b ch 1 t

    zm_min = zm[:, 0].min(zm[:, 1])
    zm_max = zm[:, 0].max(zm[:, 1])
    zm_feat = torch.stack([zm_min, zm_max], dim=1)
    zzm = torch.cat([zm, zm_feat], dim=1)  # b ch 1 t

    if not drop_z:
        left = torch.cat([ll, lp, zz], dim=2)  # b ch g t
        right = torch.cat([rl, rp, zz], dim=2)  # b ch g t
        left_mask = torch.cat([llm, lpm, zzm], dim=2)  # b ch g t
        right_mask = torch.cat([rlm, rpm, zzm], dim=2)  # b ch g t
    else:
        left = torch.cat([ll, lp], dim=2)  # b ch g t
        right = torch.cat([rl, rp], dim=2)  # b ch g t
        left_mask = torch.cat([llm, lpm], dim=2)  # b ch g t
        right_mask = torch.cat([rlm, rpm], dim=2)  # b ch g t

    x = torch.stack([left, right], dim=0)  # d b ch g t
    xm = torch.stack([left_mask, right_mask], dim=0)  # d b ch g t

    return x, xm


class EegDualPartialStackingCollator(nn.Module):
    def __init__(self, drop_z: bool = False):
        super().__init__()
        self.drop_z = drop_z

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(drop_z={self.drop_z})"

    def forward(self, eeg: Tensor, eeg_mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        output:
        eeg: (d b) (ch g) t
        eeg_mask: (d b) (ch g) t
        """
        eeg, eeg_mask = dual_partial_stack_eeg_channels(
            eeg, eeg_mask, drop_z=self.drop_z
        )
        eeg = rearrange(eeg, "d b ch g t -> (d b) (ch g) t")
        eeg_mask = rearrange(eeg_mask, "d b ch g t -> (d b) (ch g) t")
        return eeg, eeg_mask


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
    """
    左右ごとに独立してencodingする
    """

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


def horizontal_dual_stack_eeg_channels(x: Tensor, num_channels: int) -> Tensor:
    """
    spec: (b, 2 * ch, t)
    mask: (b, 2 * ch, t)

    Return:
    spec: (2, b, ch, t)
    mask: (2, b, ch, t)
    """
    assert x.shape[1] == 2 * num_channels
    left = x[:, 0:num_channels]
    right = x[:, num_channels : 2 * num_channels]
    x = torch.stack([left, right], dim=0)

    return x.contiguous()


class EegHorizontalDualStackingCollator(nn.Module):
    """
    左右ごとに独立してencodingする
    """

    def __init__(self, num_channels: int = 16):
        super().__init__()
        self.num_channels = num_channels

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    def forward(self, eeg: Tensor, eeg_mask: Tensor) -> tuple[Tensor, Tensor]:
        eeg = horizontal_dual_stack_eeg_channels(eeg, num_channels=self.num_channels)
        eeg_mask = horizontal_dual_stack_eeg_channels(
            eeg_mask, num_channels=self.num_channels
        )
        eeg = rearrange(eeg, "d b c t -> (d b) c t")
        eeg_mask = rearrange(eeg_mask, "d b c t -> (d b) c t")

        return eeg, eeg_mask


class EegDualPerChannelCollator(nn.Module):
    """
    左右、チャネルごとに独立してencodingする
    """

    def __init__(self, drop_z: bool = False):
        super().__init__()
        self.drop_z = drop_z

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(drop_z={self.drop_z})"

    def forward(self, eeg: Tensor, eeg_mask: Tensor) -> tuple[Tensor, Tensor]:
        eeg = dual_stack_eeg_channels(eeg, drop_z=self.drop_z)
        eeg_mask = dual_stack_eeg_channels(eeg_mask, drop_z=self.drop_z)
        eeg = rearrange(eeg, "d b c t -> (d c b) 1 t")
        eeg_mask = rearrange(eeg_mask, "d b c t -> (d c b) 1 t")

        return eeg, eeg_mask
