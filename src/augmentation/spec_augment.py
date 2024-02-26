import torch
import torch.nn as nn
from torch import Tensor
from torchaudio.functional import mask_along_axis, mask_along_axis_iid


def time_freq_masking(
    spec: Tensor,
    time_mask_param: int,
    freq_mask_param: int,
    iid_masks: bool = True,
    num_time_masks: int = 1,
    num_freq_masks: int = 1,
) -> Tensor:
    assert spec.ndim == 4, "Input tensor must be 4D"
    mask_fn = mask_along_axis_iid if iid_masks else mask_along_axis

    for _ in range(num_time_masks):
        spec = mask_fn(
            spec,
            mask_param=time_mask_param,
            mask_value=0.0,
            axis=3,
        )

    for _ in range(num_freq_masks):
        spec = mask_fn(
            spec,
            mask_param=freq_mask_param,
            mask_value=0.0,
            axis=2,
        )
    return spec


class NaiveTimeFreqMasking(nn.Module):
    def __init__(
        self,
        time_mask_ratio: float = 0.25,
        freq_mask_ratio: float = 0.25,
        num_time_masks: int = 1,
        num_freq_masks: int = 1,
        iid_masks: bool = True,
    ):
        super().__init__()
        self.time_mask_ratio = time_mask_ratio
        self.freq_mask_ratio = freq_mask_ratio
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks
        self.iid_masks = iid_masks

    @torch.no_grad()
    def forward(self, spec: Tensor) -> Tensor:
        B, C, F, T = spec.shape
        time_mask_param = int(T * self.time_mask_ratio)
        freq_mask_param = int(F * self.freq_mask_ratio)

        spec = time_freq_masking(
            spec,
            time_mask_param,
            freq_mask_param,
            self.iid_masks,
            self.num_time_masks,
            self.num_freq_masks,
        )

        return spec


class ChannelSyncedTimeFreqMasking(nn.Module):
    """
    EEG specに特化したMasking
    1. Batch方向はiidに
    2. channel方向は一様に

    Note: *inplace* implementation: the original tensor will be modified.
    """

    def __init__(
        self,
        time_mask_ratio: float = 0.2,
        freq_mask_ratio: float = 0.2,
        num_time_masks: int = 2,
        num_freq_masks: int = 2,
        p: float = 0.5,
    ):
        super().__init__()
        self.time_mask_ratio = time_mask_ratio
        self.freq_mask_ratio = freq_mask_ratio
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks
        self.p = p

    @torch.no_grad()
    def forward(self, spec: Tensor) -> Tensor:
        B, C, F, T = spec.shape
        time_mask_param = int(T * self.time_mask_ratio)
        freq_mask_param = int(F * self.freq_mask_ratio)
        pp = torch.rand(B)
        for i in range(B):
            if pp[i] < self.p:
                for _ in range(self.num_time_masks):
                    spec[i] = mask_along_axis(
                        spec[i], mask_param=freq_mask_param, mask_value=0.0, axis=2
                    )

                for _ in range(self.num_freq_masks):
                    spec[i] = mask_along_axis(
                        spec[i], mask_param=time_mask_param, mask_value=0.0, axis=1
                    )

        return spec
