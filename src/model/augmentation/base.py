import torch
import torch.nn as nn
from torch import Tensor


class BaseAugmentation(nn.Module):
    def __init__(self, p: float = 1.0):
        super().__init__()
        self.p = p

    def forward(self, batch: dict[str, Tensor], output: dict[str, Tensor]) -> None:
        if torch.rand(1).item() < self.p:
            self.apply(batch, output)

    def apply(self, batch: dict[str, Tensor], output: dict[str, Tensor]) -> None:
        pass


class Compose(BaseAugmentation):
    def __init__(self, augmentations: list[BaseAugmentation], p: float = 1.0):
        super().__init__(p=p)
        self.augmentations = augmentations

    def apply(self, batch: dict[str, Tensor], output: dict[str, Tensor]) -> None:
        for augmentation in self.augmentations:
            augmentation(batch, output)


class Identity(BaseAugmentation):
    def apply(self, batch: dict[str, Tensor], output: dict[str, Tensor]) -> None:
        pass
