import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor


def bg_collate_lr_channels(spec: Tensor, pad_with_mean: bool = False) -> Tensor:
    """
    左右のchannelをbatch方向に積み上げる
    Zチャネルは左右両方に入力する

    spec: (B, C, F, T)

    Return:
    spec: (2B, C, F, T)
    """
    assert spec.shape[1] == 4, f"spec shape mismatch: {spec.shape}"
    if pad_with_mean:
        pad = spec.mean(dim=1, keepdim=True)
    else:
        pad = torch.zeros_like(spec[:, 0:1, ...]).to(spec.device)

    spec_left = torch.cat([spec[:, 0:2, ...], pad], dim=1)  # (LL, LP, pad)
    spec_right = torch.cat([spec[:, 2:4, ...], pad], dim=1)  # (RL, RP, pad)
    spec = torch.cat([spec_left, spec_right], dim=0)

    return spec


def bg_fill_canvas(spec: Tensor) -> Tensor:
    """
    spec: (B, 4, F, T)

    Return:
    spec: (B, 2, F, 2T)
    """
    assert spec.shape[1] == 4, f"spec shape mismatch: {spec.shape}"
    spec_l = spec[:, :2, ...]
    spec_r = spec[:, 2:, ...]
    spec = torch.cat([spec_l, spec_r], dim=3)  # (B, 2, F, 2T)
    return spec


class BgDualStackAggregator(nn.Module):
    def __init__(self, pad_with_mean: bool = False):
        super().__init__()
        self.pad_with_mean = pad_with_mean

    def forward(self, spec: Tensor) -> Tensor:
        return bg_collate_lr_channels(spec, self.pad_with_mean)


class BgDualTilingAggregator(nn.Module):
    def __init__(self, pad_with_mean: bool = False):
        super().__init__()
        self.pad_with_mean = pad_with_mean

    def forward(self, spec: Tensor) -> Tensor:
        spec = bg_collate_lr_channels(spec, self.pad_with_mean)
        spec = rearrange(spec, "b c f t -> b 1 (c f) t")

        return spec


class BgDualCanvasAggregator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, spec: Tensor) -> Tensor:
        """
        input: (B, 18, F, T)
        output: (2B, 1, 5F, 2T)
        """
        spec = bg_fill_canvas(spec)  # (B, 2, F, 2T)

        spec = rearrange(spec, "b c f t -> (c b) 1 f t")

        return spec
