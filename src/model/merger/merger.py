import torch
import torch.nn as nn
from torch import Tensor


class MergeFreq(nn.Module):
    """
    周波数方向にマージする
    """

    def forward(
        self, spec: Tensor, mask: Tensor, bg_spec: Tensor, bg_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        input
        -----
        spec: (B, C, F0, T)
        bg_spec: (B, C, F1, T)

        returns
        -------
        spec: (B, C, F0 + F1, T)
        """
        b0, c0, _, t0 = spec.shape
        b1, c1, _, t1 = bg_spec.shape
        b2, c2, _, t2 = mask.shape
        b3, c3, _, t3 = bg_mask.shape
        assert (
            b0 == b1 and c0 == c1 and t0 == t1
        ), f"spec shape mismatch: spec={spec.shape}, bg_spec={bg_spec.shape}"
        assert (
            b2 == b3 and c2 == c3 and t2 == t3
        ), f"mask shape mismatch: mask={mask.shape}, bg_mask={bg_mask.shape}"
        spec = torch.cat([spec, bg_spec], dim=2)
        mask = torch.cat([mask, bg_mask], dim=2)

        return spec, mask
