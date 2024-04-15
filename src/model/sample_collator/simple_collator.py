import torch.nn as nn
from einops import rearrange
from torch import Tensor


class SimpleCollator(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, feature: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        feature = rearrange(feature, "b s t c -> (b s) t c")
        mask = rearrange(mask, "b s t c -> (b s) t c")
        return feature, mask
