import torch.nn as nn
import torch.nn.functional as F
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
            align_corners=False
            if mode in {"bilinear", "bicubic", "trilinear"}
            else None,
        )

    def forward(self, spec: Tensor) -> Tensor:
        spec = self.resize(spec)
        return spec


class BgPaddingTransform(nn.Module):
    def __init__(
        self,
        size: tuple[int, int],
    ):
        super().__init__()
        self.size = tuple(size)

    def forward(self, spec: Tensor) -> Tensor:
        _, _, f, t = spec.shape
        assert (
            f <= self.size[0] and t <= self.size[1]
        ), f"spec size must be smaller than {self.size}"
        pad_size = (0, self.size[1] - t, 0, self.size[0] - f)
        spec = F.pad(spec, pad_size, value=0)

        return spec
