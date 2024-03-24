import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class Mixup1d(nn.Module):
    def __init__(self, alpha: float = 10.0):
        super().__init__()
        self.alpha = alpha

    def forward(
        self, signal: Tensor, mask: Tensor, label: Tensor, weight: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, float]:
        """
        signal: b ch t
        mask: b ch t
        label: b c
        weight: b 1
        """
        lambd = np.random.beta(self.alpha, self.alpha)

        signal = lambd * signal + (1 - lambd) * signal.flip(0)
        mask = lambd * mask + (1 - lambd) * mask.flip(0)
        label = lambd * label + (1 - lambd) * label.flip(0)
        weight = lambd * weight + (1 - lambd) * weight.flip(0)

        return signal, mask, label, weight, lambd


if __name__ == "__main__":
    from einops import rearrange

    np.random.seed(0)
    torch.manual_seed(0)

    batch_size, channels, frame_length = 2, 1, 10
    num_classes = 6
    signal = torch.from_numpy(np.array([[1.0] * frame_length, [0.0] * frame_length]))
    mask = torch.from_numpy(np.array([[1.0] * frame_length, [0.0] * frame_length]))
    label = torch.from_numpy(np.array([[1.0] * num_classes, [0.0] * num_classes]))
    weight = torch.from_numpy(np.array([[1.0], [0.0]]))

    signal = rearrange(signal, "b t -> b 1 t", b=batch_size, t=frame_length)
    mask = rearrange(mask, "b t -> b 1 t", b=batch_size, t=frame_length)

    cutmix = Mixup1d()

    mixed_signal, mixed_mask, mixed_labels, mixed_weights, lambd = cutmix(
        signal, mask, label, weight
    )

    print(f"lambda: {lambd:.3f}")
    print(f"signal: {mixed_signal.detach().cpu().numpy().round(3)}")
    print(f"mask: {mixed_mask.detach().cpu().numpy().round(3)}")
    print(f"label: {mixed_labels.detach().cpu().numpy().round(3)}")
    print(f"weight: {mixed_weights.detach().cpu().numpy().round(3)}")
