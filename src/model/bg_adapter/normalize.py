import torch.nn as nn
from torch import Tensor

BG_CHANNEL_MEAN = -31.65
BG_CHANNEL_STD = 12.66


class BgConstantNormalizer(nn.Module):
    """
    固定の平均値と標準偏差でダイナミックレンジをスケールする
    """

    def __init__(
        self,
        mean: float = BG_CHANNEL_MEAN,
        std: float = BG_CHANNEL_STD,
    ):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, spec: Tensor) -> Tensor:
        spec = (spec - self.mean) / self.std

        return spec
