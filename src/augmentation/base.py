import torch
import torch.nn as nn
from torch import Tensor


class Compose(nn.Module):
    def __init__(self, transforms: list[nn.Module]):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    @torch.no_grad()
    def forward(self, spec: Tensor) -> Tensor:
        for transform in self.transforms:
            spec = transform(spec)
        return spec
