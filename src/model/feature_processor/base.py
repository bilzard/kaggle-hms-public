import torch.nn as nn
from torch import Tensor


class BaseFeatureProcessor(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

    @property
    def out_channels(self) -> int:
        raise NotImplementedError

    def forward(self, inputs: dict[str, Tensor]) -> Tensor:
        raise NotImplementedError


class IdentityFeatureProcessor(BaseFeatureProcessor):
    def __init__(self, in_channels: int):
        super().__init__(in_channels=in_channels)

    @property
    def out_channels(self) -> int:
        return self.in_channels

    def forward(self, inputs: dict[str, Tensor]) -> Tensor:
        return inputs["spec"]
