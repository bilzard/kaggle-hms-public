import numpy as np
import torch
from torch import Tensor

from src.model.augmentation.base import BaseAugmentation


def cutmix_1d(
    alpha: float,
    signal: Tensor,
    mask: Tensor,
    label: Tensor,
    weight: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, float]:
    """
    signal: b ch t
    mask: b ch t
    label: b k c
    weight: b k
    """
    _, _, frame_length = signal.shape

    lambd = np.random.beta(alpha, alpha)

    cut_len = int(frame_length * (1 - lambd))
    start = np.random.randint(0, frame_length - cut_len + 1)
    end = start + cut_len

    signal[:, :, start:end] = signal.flip(0)[:, :, start:end]
    mask[:, :, start:end] = mask.flip(0)[:, :, start:end]
    label = lambd * label + (1 - lambd) * label.flip(0)
    weight = lambd * weight + (1 - lambd) * weight.flip(0)

    return signal, mask, label, weight, lambd


class IdealCutmix1d(BaseAugmentation):
    def __init__(self, p: float, alpha: float = 5.0, prob_threshold: float = 0.8):
        """
        ラベルの最大確率が閾値以上のサンプル(ideal)同士のみをほぼ半々の確率で混ぜ合わせる
        proto/edgeのクラスを擬似的に生成するのが目的。
        """
        super().__init__(p=p)
        self.alpha = alpha
        self.prob_threshold = prob_threshold

    def apply(self, batch: dict[str, Tensor], output: dict[str, Tensor]):
        signal = output["eeg"]
        mask = output["eeg_mask"]
        label = batch["label"]
        weight = batch["weight"]

        ideal_indices = torch.max(label[:, 0], dim=-1).values > self.prob_threshold

        if sum(ideal_indices) <= 1:
            return

        signal_ideal = signal[ideal_indices]
        mask_ideal = mask[ideal_indices]
        label_ideal = label[ideal_indices]
        weight_ideal = weight[ideal_indices]

        signal_normal = signal[~ideal_indices]
        mask_normal = mask[~ideal_indices]
        label_normal = label[~ideal_indices]
        weight_normal = weight[~ideal_indices]

        signal_ideal, mask_ideal, label_ideal, weight_ideal, lambd = cutmix_1d(
            self.alpha, signal_ideal, mask_ideal, label_ideal, weight_ideal
        )

        mixed_signal = torch.cat([signal_normal, signal_ideal], dim=0)
        mixed_mask = torch.cat([mask_normal, mask_ideal], dim=0)
        mixed_labels = torch.cat([label_normal, label_ideal], dim=0)
        mixed_weights = torch.cat([weight_normal, weight_ideal], dim=0)

        output["eeg"] = mixed_signal
        output["eeg_mask"] = mixed_mask
        batch["label"] = mixed_labels
        batch["weight"] = mixed_weights
        output["lambda"] = lambd  # type: ignore


if __name__ == "__main__":
    from einops import rearrange

    def generate_samples(batch_size=2, channels=1, frame_length=10, num_classes=6):
        signal = torch.arange(batch_size * frame_length).float()
        mask = torch.arange(batch_size * frame_length).float()
        label = torch.randn(batch_size, 2, num_classes).softmax(dim=-1)
        weight = torch.from_numpy(np.array([[1.0], [0.0]]))

        signal = rearrange(signal, "(b t) -> b 1 t", b=batch_size, t=frame_length)
        mask = rearrange(mask, "(b t) -> b 1 t", b=batch_size, t=frame_length)

        return signal, mask, label, weight

    print("*" * 80)
    print("* Test Cutmix1d")
    print("*" * 80)
    np.random.seed(0)
    torch.manual_seed(0)

    aug = IdealCutmix1d(p=0.5, alpha=20, prob_threshold=0.3)
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
