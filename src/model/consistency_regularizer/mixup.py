import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor


def auto_unsqueeze_to_match(tensor: Tensor, target: Tensor) -> Tensor:
    """
    指定された形状に一致するようにテンソルに自動的にunsqueezeを適用する。

    Args:
        target_shape (tuple): 目標とするテンソルの形状。
        tensor (torch.Tensor): 元のテンソル。

    Returns:
        torch.Tensor: 形状が指定された形状に一致するように変更されたテンソル。
    """
    assert (
        tensor.dim() <= target.dim()
    ), f"target dim {target.dim()} is smaller than tensor dim {tensor.dim()}"
    dim_diff = target.dim() - tensor.dim()

    for _ in range(dim_diff):
        tensor = tensor.unsqueeze(-1)

    match target.dim():
        case 4:
            B, C, F, T = target.shape
            tensor = tensor.expand(B, C, F, T)

        case 3:
            B, C, T = target.shape
            tensor = tensor.expand(B, C, T)

        case 2:
            B, C = target.shape
            tensor = tensor.expand(B, C)

        case 1:
            pass

        case _:
            raise ValueError(f"target dim {target.dim()} is not supported")

    return tensor


def apply_mixup(
    spec: Tensor,
    spec_mask: Tensor,
    eeg: Tensor,
    eeg_mask: Tensor,
    label: Tensor,
    weight: Tensor,
    alpha: float = 0.25,
    p: float = 1.0,
):
    if np.random.rand() > p:
        return spec, spec_mask, eeg, eeg_mask, label, weight

    B = label.shape[0]

    lam0 = np.random.beta(alpha, alpha, size=B)
    lam0 = torch.from_numpy(lam0).to(spec.dtype).to(spec.device)
    idx = torch.randperm(B).to(spec.device)

    lam = auto_unsqueeze_to_match(lam0, spec)
    spec_mask_mixed = lam * spec_mask + (1 - lam) * spec_mask[idx]
    spec_mixed = lam * spec + (1 - lam) * spec[idx]

    lam = auto_unsqueeze_to_match(lam0, eeg)
    eeg_mask_mixed = lam * eeg_mask + (1 - lam) * eeg_mask[idx]
    eeg_mixed = lam * eeg + (1 - lam) * eeg[idx]

    lam = auto_unsqueeze_to_match(lam0, weight)
    weight_mixed = lam * weight + lam[idx] * weight[idx]

    lam = auto_unsqueeze_to_match(lam0, label)
    label_mixed = lam * weight * label + lam[idx] * weight[idx] * label[idx]
    label_mixed = label_mixed / label_mixed.sum(dim=1, keepdim=True)

    return (
        spec_mixed,
        spec_mask_mixed,
        eeg_mixed,
        eeg_mask_mixed,
        label_mixed,
        weight_mixed,
    )


class MixUp(nn.Module):
    def __init__(
        self,
        alpha: float = 0.25,
        p: float = 0.3,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.p = p

    def __repr__(self):
        return f"{self.__class__.__name__}({self.alpha=}, {self.p=}, {self.eps=})"

    def forward(
        self,
        spec: Tensor,
        spec_mask: Tensor,
        eeg: Tensor,
        eeg_mask: Tensor,
        label: Tensor,
        weight: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        B = label.shape[0]
        B_spec = spec.shape[0]
        d = B_spec // B

        if d != 1:
            spec = rearrange(spec, "(d b) c f t -> b (d c) f t", d=d)
            spec_mask = rearrange(spec_mask, "(d b) c f t -> b (d c) f t", d=d)

        spec, spec_mask, eeg, eeg_mask, label, weight = apply_mixup(
            spec=spec,
            spec_mask=spec_mask,
            eeg=eeg,
            eeg_mask=eeg_mask,
            label=label,
            weight=weight,
            alpha=self.alpha,
            p=self.p,
        )

        if d != 1:
            spec = rearrange(spec, "b (d c) f t -> (d b) c f t", d=d)
            spec_mask = rearrange(spec_mask, "b (d c) f t -> (d b) c f t", d=d)

        return spec, spec_mask, eeg, eeg_mask, label, weight


class MixUpFixedRatio(nn.Module):
    """
    batch内の一定の割合に対して確定的にmixupする
    """

    def __init__(
        self,
        alpha: float = 0.25,
        p: float = 0.3,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.p = p

    def __repr__(self):
        return f"{self.__class__.__name__}({self.alpha=}, {self.p=}, {self.eps=})"

    def forward(
        self,
        spec: Tensor,
        spec_mask: Tensor,
        eeg: Tensor,
        eeg_mask: Tensor,
        label: Tensor,
        weight: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        clean sampleどうし、noisy sampleどうしをそれぞれmixupする。
        """
        B = label.shape[0]
        B_spec = spec.shape[0]
        d = B_spec // B

        if d != 1:
            spec = rearrange(spec, "(d b) c f t -> b (d c) f t", d=d)
            spec_mask = rearrange(spec_mask, "(d b) c f t -> b (d c) f t", d=d)

        num_mixed = int(B * self.p)
        mixed_idxs = torch.randperm(B) <= num_mixed
        (
            spec_mixed,
            spec_mask_mixed,
            eeg_mixed,
            eeg_mask_mixed,
            label_mixed,
            weight_mixed,
        ) = apply_mixup(
            spec=spec[mixed_idxs],
            spec_mask=spec_mask[mixed_idxs],
            eeg=eeg[mixed_idxs],
            eeg_mask=eeg_mask[mixed_idxs],
            label=label[mixed_idxs],
            weight=weight[mixed_idxs],
            alpha=self.alpha,
            p=1.0,
        )

        spec = torch.cat([spec_mixed, spec[~mixed_idxs]], dim=0)
        spec_mask = torch.cat([spec_mask_mixed, spec_mask[~mixed_idxs]], dim=0)
        eeg = torch.cat([eeg_mixed, eeg[~mixed_idxs]], dim=0)
        eeg_mask = torch.cat([eeg_mask_mixed, eeg[~mixed_idxs]], dim=0)
        label = torch.cat([label_mixed, label[~mixed_idxs]], dim=0)
        weight = torch.cat([weight_mixed, weight[~mixed_idxs]], dim=0)

        if d != 1:
            spec = rearrange(spec, "b (d c) f t -> (d b) c f t", d=d)
            spec_mask = rearrange(spec_mask, "b (d c) f t -> (d b) c f t", d=d)

        return spec, spec_mask, eeg, eeg_mask, label, weight


class MixUpPerGroup(nn.Module):
    def __init__(
        self,
        alpha: float = 0.25,
        p_clean: float = 0.3,
        p_noisy: float = 0.0,
        weight_threshold: float = 0.3,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.p_clean = p_clean
        self.p_noisy = p_noisy
        self.weight_threshold = weight_threshold

    def __repr__(self):
        return f"{self.__class__.__name__}({self.alpha=}, {self.p_clean=}, {self.p_noisy=}, {self.weight_threshold=}, {self.eps=})"

    def forward(
        self,
        spec: Tensor,
        spec_mask: Tensor,
        eeg: Tensor,
        eeg_mask: Tensor,
        label: Tensor,
        weight: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        clean sampleどうし、noisy sampleどうしをそれぞれmixupする。
        """
        B = label.shape[0]
        B_spec = spec.shape[0]
        d = B_spec // B

        if d != 1:
            spec = rearrange(spec, "(d b) c f t -> b (d c) f t", d=d)
            spec_mask = rearrange(spec_mask, "(d b) c f t -> b (d c) f t", d=d)

        clean_idxs = weight[:, 0] > self.weight_threshold
        (
            spec_clean,
            spec_mask_clean,
            eeg_clean,
            eeg_mask_clean,
            label_clean,
            weight_clean,
        ) = apply_mixup(
            spec=spec[clean_idxs],
            spec_mask=spec_mask[clean_idxs],
            eeg=eeg[clean_idxs],
            eeg_mask=eeg_mask[clean_idxs],
            label=label[clean_idxs],
            weight=weight[clean_idxs],
            alpha=self.alpha,
            p=self.p_clean,
        )
        (
            spec_noisy,
            spec_mask_noisy,
            eeg_noisy,
            eeg_mask_noisy,
            label_noisy,
            weight_noisy,
        ) = apply_mixup(
            spec=spec[~clean_idxs],
            spec_mask=spec_mask[~clean_idxs],
            eeg=eeg[~clean_idxs],
            eeg_mask=eeg_mask[~clean_idxs],
            label=label[~clean_idxs],
            weight=weight[~clean_idxs],
            alpha=self.alpha,
            p=self.p_noisy,
        )

        spec = torch.cat([spec_clean, spec_noisy], dim=0)
        spec_mask = torch.cat([spec_mask_clean, spec_mask_noisy], dim=0)
        eeg = torch.cat([eeg_clean, eeg_noisy], dim=0)
        eeg_mask = torch.cat([eeg_mask_clean, eeg_mask_noisy], dim=0)
        label = torch.cat([label_clean, label_noisy], dim=0)
        weight = torch.cat([weight_clean, weight_noisy], dim=0)

        if d != 1:
            spec = rearrange(spec, "b (d c) f t -> (d b) c f t", d=d)
            spec_mask = rearrange(spec_mask, "b (d c) f t -> (d b) c f t", d=d)

        return spec, spec_mask, eeg, eeg_mask, label, weight
