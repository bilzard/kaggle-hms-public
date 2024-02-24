import numpy as np
import torch

from src.constant import PROBE2IDX
from src.transform.base import BaseTransform

LR_CORRESPONDENCE = [
    ("Fp1", "Fp2"),
    ("F3", "F4"),
    ("C3", "C4"),
    ("P3", "P4"),
    ("F7", "F8"),
    ("T3", "T4"),
    ("T5", "T6"),
    ("O1", "O2"),
]
FR_CORRESPONDENCE = [
    ("Fp1", "O1"),
    ("Fp2", "O2"),
    ("F7", "T5"),
    ("F3", "P3"),
    ("Fz", "Pz"),
    ("F4", "P4"),
    ("F8", "T6"),
]
CENTRAL_CHANNELS = ["Fz", "Cz", "Pz"]


def _swap_channels(
    feature: np.ndarray,
    mask: np.ndarray,
    correspondences: list,
) -> tuple[np.ndarray, np.ndarray]:
    """
    対応するチャネルを入れ替える
    feature: (num_samples, num_frames, num_features)
    """
    feature, mask = feature.copy(), mask.copy()
    for src, dst in correspondences:
        src_idx, dst_idx = PROBE2IDX[src], PROBE2IDX[dst]
        feature[..., [src_idx, dst_idx]] = feature[..., [dst_idx, src_idx]]
        mask[..., [src_idx, dst_idx]] = mask[..., [dst_idx, src_idx]]

    return feature, mask


def swap_lr(
    feature: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    左右のチャネルを入れ替える
    feature: (num_samples, num_frames, num_features)
    """
    return _swap_channels(feature, mask, LR_CORRESPONDENCE)


def swap_fr(
    feature: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    前頭側と後頭側のチャネルの順番を入れ替える
    feature: (num_samples, num_frames, num_features)
    """
    return _swap_channels(feature, mask, FR_CORRESPONDENCE)


def channel_permutation(
    feature: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    チャンネルの順番をランダムに入れ替える
    feature: (num_samples, num_frames, num_features)
    """
    feature, mask = feature.copy(), mask.copy()
    left_idxs, right_idxs = [], []
    for left, right in LR_CORRESPONDENCE:
        left_idx, right_idx = PROBE2IDX[left], PROBE2IDX[right]
        left_idxs.append(left_idx)
        right_idxs.append(right_idx)

    left_idxs = np.array(left_idxs)
    right_idxs = np.array(right_idxs)

    perm = torch.randperm(len(left_idxs)).numpy()

    feature[..., left_idxs] = feature[..., left_idxs[perm]]
    feature[..., right_idxs] = feature[..., right_idxs[perm]]
    mask[..., left_idxs] = mask[..., left_idxs[perm]]
    mask[..., right_idxs] = mask[..., right_idxs[perm]]

    central_idxs = np.array([PROBE2IDX[ch] for ch in CENTRAL_CHANNELS])
    perm = torch.randperm(len(central_idxs)).numpy()
    feature[..., central_idxs] = feature[..., central_idxs[perm]]
    mask[..., central_idxs] = mask[..., central_idxs[perm]]

    return feature, mask


class SwapLr(BaseTransform):
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
        return swap_lr(feature, mask)


class SwapFr(BaseTransform):
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
        return swap_fr(feature, mask)


class ChannelPermutation(BaseTransform):
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
        return channel_permutation(feature, mask)


if __name__ == "__main__":
    for transform_cls in [SwapLr, SwapFr, ChannelPermutation]:
        print("*" * 40)
        print(f"** {transform_cls.__name__} **")
        print("*" * 40)
        transform = transform_cls(p=1.0)
        feat = np.array(list(PROBE2IDX.keys()))[np.newaxis, np.newaxis, :]
        mask = np.array(list(PROBE2IDX.keys()))[np.newaxis, np.newaxis, :]
        mask = np.array(list(PROBE2IDX.keys()))[np.newaxis, np.newaxis, :]

        feat_aug, mask_aug = transform(feat, mask)
        print(feat.shape)
        print("* feature *")
        for ch in range(feat.shape[-1]):
            print(f"{feat[..., ch].item()} -> {feat_aug[..., ch].item()}")

        assert (mask_aug == feat_aug).all()
