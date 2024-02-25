import torch.nn as nn
from torch import Tensor


class BgResizeTransform(nn.Module):
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
            align_corners=False,
        )

    def forward(self, spec: Tensor) -> Tensor:
        spec = self.resize(spec)
        return spec
