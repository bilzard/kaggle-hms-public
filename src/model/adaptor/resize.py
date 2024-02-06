import torch
import torch.nn as nn


class Resize(nn.Module):
    def __init__(self, scale_factor: tuple[float, float], mode: str = "bilinear"):
        super().__init__()
        self.scale_factor = (scale_factor[0], scale_factor[1])
        self.resize = nn.Upsample(
            scale_factor=self.scale_factor, mode=mode, align_corners=False
        )
        self.resize_mask = nn.Upsample(
            scale_factor=(1, self.scale_factor[1]), mode=mode, align_corners=False
        )

    def forward(
        self, spec: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spec = self.resize(spec)
        mask = self.resize_mask(mask)
        return spec, mask
