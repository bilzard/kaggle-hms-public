import numpy as np
import torch
from torch import Tensor

from src.model.augmentation.base import BaseAugmentation


def cut_mix_same_class_for_spec(
    spec: Tensor,
    mask: Tensor,
    label: Tensor,
    prob_thr: float = 0.75,
    min_size: int = 10,  # 4sec
    max_size: int = 50,  # 20sec
    max_frames: int = 125,  # 50sec
) -> tuple[Tensor, Tensor]:
    """
    spec: b ch f t
    mask: b ch f t
    label: b 6
    """
    spec, mask, label = spec.clone(), mask.clone(), label.clone()
    assert label.shape[1] == 6

    for c in range(6):
        idxs = torch.nonzero(label[:, c] >= prob_thr, as_tuple=False)

        if len(idxs) < 2:
            continue

        spec_org, mask_org = spec[idxs], mask[idxs]
        idxs_shuffled = torch.randperm(len(idxs))

        size = np.random.randint(min_size, max_size)
        if np.random.choice([True, False]):
            # swap left 20sec
            start = np.random.randint(0, max_size - size)
            end = start + size
        else:
            # swap right 20sec
            end = np.random.randint(max_frames - max_size + size, max_frames)
            start = end - size

        spec[idxs, ..., start:end] = spec_org[idxs_shuffled, ..., start:end]
        mask[idxs, ..., start:end] = mask_org[idxs_shuffled, ..., start:end]

    return spec, mask


class CutMixSameClassForSpec(BaseAugmentation):
    def __init__(
        self,
        p: float,
        prob_thr: float = 0.75,
        min_size: int = 25,  # 10sec
        max_size: int = 50,  # 20sec
        max_frames: int = 125,  # 50sec
    ):
        super().__init__(p=p)
        self.prob_thr = prob_thr
        self.min_size = min_size
        self.max_size = max_size
        self.max_frames = max_frames

    def apply(self, batch: dict[str, Tensor], output: dict[str, Tensor]) -> None:
        output["spec"], output["spec_mask"] = cut_mix_same_class_for_spec(
            output["spec"],
            output["spec_mask"],
            batch["label"],
            prob_thr=self.prob_thr,
            min_size=self.min_size,
            max_size=self.max_size,
            max_frames=self.max_frames,
        )
