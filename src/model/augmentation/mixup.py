import numpy as np
import torch
from torch import Tensor

from src.model.augmentation.base import BaseAugmentation
from src.model.consistency_regularizer.mixup import auto_unsqueeze_to_match


def mixup(
    spec: Tensor,
    spec_mask: Tensor,
    eeg: Tensor,
    eeg_mask: Tensor,
    label: Tensor,
    weight: Tensor,
    alpha: float = 0.25,
    norm_by_votes: bool = False,
):
    B = label.shape[0]

    lam0 = np.random.beta(alpha, alpha, size=B)
    lam0 = torch.from_numpy(lam0).to(spec.dtype).to(spec.device)
    shuffled_idxs = torch.randperm(B).to(spec.device)

    lam = auto_unsqueeze_to_match(lam0, spec)
    spec_mask_mixed = lam * spec_mask + (1 - lam) * spec_mask[shuffled_idxs]
    spec_mixed = lam * spec + (1 - lam) * spec[shuffled_idxs]

    lam = auto_unsqueeze_to_match(lam0, eeg)
    eeg_mask_mixed = lam * eeg_mask + (1 - lam) * eeg_mask[shuffled_idxs]
    eeg_mixed = lam * eeg + (1 - lam) * eeg[shuffled_idxs]

    lam = auto_unsqueeze_to_match(lam0, weight)
    weight_mixed = lam * weight + lam[shuffled_idxs] * weight[shuffled_idxs]

    lam = auto_unsqueeze_to_match(lam0, label)
    if norm_by_votes:
        label_mixed = (
            lam * weight * label
            + lam[shuffled_idxs] * weight[shuffled_idxs] * label[shuffled_idxs]
        )
    else:
        label_mixed = lam * label + lam[shuffled_idxs] * label[shuffled_idxs]

    # normalize label to sum to 1
    label_mixed = label_mixed / label_mixed.sum(dim=1, keepdim=True)

    return (
        spec_mixed,
        spec_mask_mixed,
        eeg_mixed,
        eeg_mask_mixed,
        label_mixed,
        weight_mixed,
    )


class Mixup(BaseAugmentation):
    def __init__(
        self,
        p: float,
        alpha: float = 0.25,
        norm_by_votes: bool = False,
    ):
        super().__init__(p=p)
        self.alpha = alpha
        self.norm_by_votes = norm_by_votes

    def apply(self, batch: dict[str, Tensor], output: dict[str, Tensor]) -> None:
        (
            output["spec"],
            output["spec_mask"],
            output["eeg"],
            output["eeg_mask"],
            batch["label"],
            batch["weight"],
        ) = mixup(
            output["spec"],
            output["spec_mask"],
            output["eeg"],
            output["eeg_mask"],
            batch["label"],
            batch["weight"],
            alpha=self.alpha,
            norm_by_votes=self.norm_by_votes,
        )
