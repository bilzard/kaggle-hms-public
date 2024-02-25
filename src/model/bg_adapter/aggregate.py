import torch
import torch.nn as nn
from torch import Tensor


def bg_collate_lr_channels(spec: Tensor) -> Tensor:
    """
    左右のchannelをbatch方向に積み上げる
    Zチャネルは左右両方に入力する

    spec: (B, C, F, T)

    Return:
    spec: (2 * B, C, F, T)


    """
    assert spec.shape[1] == 4, f"spec shape mismatch: {spec.shape}"
    pad = torch.zeros_like(spec[:, 0:1, ...]).to(spec.device)

    spec_left = torch.cat([spec[:, 0:2, ...], pad], dim=1)  # (LL, LP, pad)
    spec_right = torch.cat([spec[:, 2:4, ...], pad], dim=1)  # (RL, RP, pad)
    spec = torch.cat([spec_left, spec_right], dim=0)

    return spec


class BgDualStackAggregator(nn.Module):
    def forward(self, spec: Tensor) -> Tensor:
        return bg_collate_lr_channels(spec)
