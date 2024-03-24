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


if __name__ == "__main__":
    print("*" * 80)
    print("* Test BaseAugmentation")
    print("*" * 80)

    class CustomAugmentation(BaseAugmentation):
        def apply(self, batch: dict[str, Tensor], output: dict[str, Tensor]) -> None:
            print("CustomAugmentation applied")

    aug = CustomAugmentation(p=0.5)

    for i in range(10):
        print(f"iter: {i}")
        batch = dict()
        output = dict()
        aug(batch, output)

    print("*" * 80)
    print("* Test Compose")
    print("*" * 80)

    class AugmentationA(BaseAugmentation):
        def apply(self, batch: dict[str, Tensor], output: dict[str, Tensor]) -> None:
            print("A is applied")

    class AugmentationB(BaseAugmentation):
        def apply(self, batch: dict[str, Tensor], output: dict[str, Tensor]) -> None:
            print("B is applied")

    aug = Compose([AugmentationA(p=0.5), AugmentationB(p=0.5)])

    for i in range(10):
        print(f"iter: {i}")
        batch = dict()
        output = dict()
        aug(batch, output)
