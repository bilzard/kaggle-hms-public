import numpy as np
import torch

from src.transform.base import BaseTransform


def reverse_sequence(
    feature: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    feature: (num_frames, num_features)
    """
    feature = feature[..., ::-1, :].copy()
    mask = mask[..., ::-1, :].copy()

    return feature, mask


def cutout_1d(
    feature: np.ndarray,
    mask: np.ndarray,
    max_length: int,
    num_cutouts=4,
    cutout_mask: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    feature: (num_frames, num_features)
    """
    feature, mask = feature.copy(), mask.copy()

    num_steps = feature.shape[-2]
    for _ in range(num_cutouts):
        length = int(torch.randint(0, max_length + 1, (1,)).item())
        length = min(length, num_steps)

        start = int(torch.randint(0, num_steps - length, (1,)).item())
        end = start + length

        feature[..., start:end, :] = 0
        if cutout_mask:
            mask[..., start:end, :] = 0

    return feature, mask


def random_shift(
    feature: np.ndarray,
    mask: np.ndarray,
    max_shift_sec: int = 20,
    sampling_rate: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """
    feature: (num_frames, num_features)
    """
    feature, mask = feature.copy(), mask.copy()
    num_frames = feature.shape[0]

    max_shift = max_shift_sec * sampling_rate
    feature = np.pad(feature, ((max_shift, max_shift), (0, 0)), mode="constant")
    mask = np.pad(mask, ((max_shift, max_shift), (0, 0)), mode="constant")

    shift = int(torch.randint(0, 2 * max_shift + 1, (1,)).item())
    feature = feature[shift : shift + num_frames, :]
    mask = mask[shift : shift + num_frames, :]

    return feature, mask


class ReverseSequence(BaseTransform):
    def __init__(
        self,
        p: float = 0.5,
    ):
        super().__init__(
            params=dict(),
            p=p,
        )

    def apply(
        self, feature: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return reverse_sequence(feature, mask)


class Cutout1d(BaseTransform):
    def __init__(
        self,
        max_length: int = 10,
        num_cutouts: int = 4,
        p: float = 0.5,
    ):
        super().__init__(
            params=dict(max_length=max_length, num_cutouts=num_cutouts),
            p=p,
        )
        self.max_length = max_length
        self.num_cutouts = num_cutouts

    def apply(
        self, feature: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return cutout_1d(feature, mask, self.max_length, self.num_cutouts)


class RandomShift(BaseTransform):
    def __init__(
        self,
        sampling_rate: int = 40,
        max_shift_sec: int = 20,
        p: float = 0.5,
    ):
        super().__init__(
            params=dict(sampling_rate=sampling_rate, max_shift_sec=max_shift_sec),
            p=p,
        )
        self.sampling_rate = sampling_rate
        self.max_shift_sec = max_shift_sec

    def apply(
        self, feature: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return random_shift(
            feature,
            mask,
            max_shift_sec=self.max_shift_sec,
            sampling_rate=self.sampling_rate,
        )


if __name__ == "__main__":
    for transform in [
        ReverseSequence(p=1.0),
        Cutout1d(p=1.0),
        RandomShift(p=1.0, max_shift_sec=10, sampling_rate=1),
    ]:
        print("*" * 40)
        print(f"** {type(transform).__name__} **")
        print("*" * 40)

        num_channels = 3
        num_frames = 30
        feat = np.arange(num_frames).reshape(-1, 1).repeat(num_channels, axis=1)
        mask = np.ones_like(feat)
        for i in range(num_channels):
            mask[i * 10 : (i + 1) * 10, i] = 0.0
        feat_aug, mask_aug = transform(feat, mask)
        print(feat.shape)
        print("** feature **")
        print(f"* original:\n{feat}")
        print(f"* augmented:\n{feat_aug}")
        print("** mask **")
        print(f"* original:\n{mask}")
        print(f"* augmented:\n{mask_aug}")
