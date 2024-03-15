import torch.nn as nn
from torch import Tensor


class Compose(nn.Module):
    def __init__(self, targets: list[nn.Module]):
        super().__init__()
        self.targets = targets

    def __repl__(self) -> str:
        adapter_classes = [adapter.__class__.__name__ for adapter in self.targets]
        adapter_classes = ", ".join(adapter_classes)
        return f"{self.__class__.__name__}(adapters={adapter_classes})"

    def forward(self, eeg: Tensor, eeg_mask: Tensor) -> tuple[Tensor, Tensor]:
        for adapter in self.targets:
            eeg, eeg_mask = adapter(eeg, eeg_mask)

        return eeg, eeg_mask
