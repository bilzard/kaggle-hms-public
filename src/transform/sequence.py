import numpy as np

from src.transform.base import BaseTransform


def reverse_sequence(
    feature: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    feature: (num_samples, num_frames, num_features)
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
    feature: (n_samples, num_frames, num_features)
    """
    feature, mask = feature.copy(), mask.copy()

    num_steps = feature.shape[-2]
    for _ in range(num_cutouts):
        length = np.random.randint(0, max_length + 1)
        length = min(length, num_steps)

        start = np.random.randint(0, num_steps - length)
        end = start + length

        feature[..., start:end, :] = 0
        if cutout_mask:
            mask[..., start:end, :] = 0

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


if __name__ == "__main__":
    for transform_cls in [ReverseSequence, Cutout1d]:
        print("*" * 40)
        print(f"** {transform_cls.__name__} **")
        print("*" * 40)
        transform = transform_cls(p=1.0)

        num_channels = 5
        num_frames = 50
        feat = np.arange(num_frames).reshape(1, -1, 1).repeat(num_channels, axis=-1)
        mask = np.ones_like(feat)
        for i in range(num_channels):
            mask[..., i * 10 : (i + 1) * 10, i] = 0.0
        feat_aug, mask_aug = transform(feat, mask)
        print(feat.shape)
        print("** feature **")
        print(f"* original:\n{feat}")
        print(f"* augmented:\n{feat_aug}")
        print("** mask **")
        print(f"* original:\n{mask}")
        print(f"* augmented:\n{mask_aug}")
