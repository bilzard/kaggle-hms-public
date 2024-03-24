import numpy as np
import torch
from torch import Tensor

from src.model.augmentation.base import BaseAugmentation


def mixup_1d(
    alpha: float, signal: Tensor, mask: Tensor, label: Tensor, weight: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor, float]:
    """
    signal: b ch t
    mask: b ch t
    label: b k c
    weight: b k
    """
    lambd = np.random.beta(alpha, alpha)

    signal = lambd * signal + (1 - lambd) * signal.flip(0)
    mask = lambd * mask + (1 - lambd) * mask.flip(0)
    label = lambd * label + (1 - lambd) * label.flip(0)
    weight = lambd * weight + (1 - lambd) * weight.flip(0)

    return signal, mask, label, weight, lambd


class Mixup1d(BaseAugmentation):
    def __init__(self, p: float, alpha: float = 10.0):
        super().__init__(p=p)
        self.alpha = alpha

    def apply(self, batch: dict[str, Tensor], output: dict[str, Tensor]):
        signal = output["eeg"]
        mask = output["eeg_mask"]
        label = batch["label"]
        weight = batch["weight"]

        mixed_signal, mixed_mask, mixed_labels, mixed_weights, lambd = mixup_1d(
            self.alpha, signal, mask, label, weight
        )
        output["eeg"] = mixed_signal
        output["eeg_mask"] = mixed_mask
        batch["label"] = mixed_labels
        batch["weight"] = mixed_weights
        output["lambda"] = lambd  # type: ignore


if __name__ == "__main__":
    from einops import rearrange

    def generate_samples(batch_size=2, channels=1, frame_length=10, num_classes=6):
        signal = torch.from_numpy(
            np.array([[1.0] * frame_length, [0.0] * frame_length])
        )
        mask = torch.from_numpy(np.array([[1.0] * frame_length, [0.0] * frame_length]))
        label = torch.from_numpy(
            np.array([[[1.0] * num_classes], [[0.0] * num_classes]])
        )
        weight = torch.from_numpy(np.array([[1.0], [0.0]]))

        signal = rearrange(signal, "b t -> b 1 t", b=batch_size, t=frame_length)
        mask = rearrange(mask, "b t -> b 1 t", b=batch_size, t=frame_length)
        return signal, mask, label, weight

    print("*" * 80)
    print("* Test Mixup1d")
    print("*" * 80)

    np.random.seed(0)
    torch.manual_seed(0)

    alpha = 2.0
    aug = Mixup1d(p=0.5, alpha=alpha)

    for i in range(10):
        print("-" * 80)
        print(f"iter: {i}")
        print("-" * 80)
        signal, mask, label, weight = generate_samples()
        batch = dict(label=label, weight=weight)
        output = dict(eeg=signal, eeg_mask=mask)
        lambd = aug(batch, output)

        if "lambda" in output:
            print(f"lambda: {output['lambda']:.3f}")
        print(f"signal: {output['eeg'].detach().cpu().numpy().round(3)}")
        print(f"mask: {output['eeg_mask'].detach().cpu().numpy().round(3)}")
        print(f"label: {batch['label'].detach().cpu().numpy().round(3)}")
        print(f"weight: {batch['weight'].detach().cpu().numpy().round(3)}")
