import torch.nn as nn
from torch import Tensor


class ResizeTransform(nn.Module):
    def __init__(self, scale_factor: tuple[float, float], mode: str = "bilinear"):
        super().__init__()
        self.scale_factor = (scale_factor[0], scale_factor[1])
        self.resize = nn.Upsample(
            scale_factor=self.scale_factor, mode=mode, align_corners=False
        )
        self.resize_mask = nn.Upsample(
            scale_factor=self.scale_factor, mode=mode, align_corners=False
        )

    def forward(self, spec: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        spec = self.resize(spec)
        mask = self.resize_mask(mask)
        return spec, mask
