import torch
import torch.nn as nn
from torch import Tensor


class ResizeTransform(nn.Module):
    def __init__(
        self,
        size: tuple[int, int] | None = None,
        scale_factor: tuple[float, float] | None = None,
        mode: str = "bilinear",
    ):
        super().__init__()
        assert (
            size is not None or scale_factor is not None
        ), "size or scale_factor must be specified"
        self.size = tuple(size) if size is not None else None
        self.scale_factor = tuple(scale_factor) if scale_factor is not None else None
        self.resize = nn.Upsample(
            size=self.size,
            scale_factor=self.scale_factor,
            mode=mode,
            align_corners=False
            if mode in {"bilinear", "bicubic", "trilinear"}
            else None,
        )
        self.resize_mask = nn.Upsample(
            size=self.size,
            scale_factor=self.scale_factor,
            mode=mode,
            align_corners=False
            if mode in {"bilinear", "bicubic", "trilinear"}
            else None,
        )

    @torch.no_grad()
    def forward(self, spec: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        spec = self.resize(spec)
        mask = self.resize_mask(mask)
        return spec, mask


class TimeCroppingTransform(nn.Module):
    def __init__(
        self,
        start: int,
        size: int,
    ):
        super().__init__()
        self.start = start
        self.size = size

    @torch.no_grad()
    def forward(self, spec: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        spec: b c f t
        mask: b c t
        """
        spec = spec[..., self.start : self.start + self.size]
        mask = mask[..., self.start : self.start + self.size]
        return spec, mask
