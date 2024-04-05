import torch
import torch.nn as nn
import torchaudio.functional as AF
from torch import Tensor

from src.constant import PROBE2IDX, PROBE_GROUPS


class ChannelCollator(nn.Module):
    def __init__(
        self,
        sampling_rate: int = 200,
        cutoff_freqs: tuple[float, float] = (0.5, 50),
        reject_freq: float | None = None,
        apply_mask: bool = True,
        probe_groups: dict[str, list[str]] = PROBE_GROUPS,
        down_sample_rate: int = 5,
        down_sample_after_filter: bool = False,
    ):
        super().__init__()

        self.sampling_rate = sampling_rate
        self.cutoff_freqs = cutoff_freqs
        self.apply_mask = apply_mask
        self.probe_groups = probe_groups
        self.down_sample_rate = down_sample_rate
        self.down_sample_after_filter = down_sample_after_filter
        self.reject_freq = reject_freq

    def __repr__(self):
        return f"""{self.__class__.__name__}(
            sampling_rate={self.sampling_rate},
            cutoff_freqs={self.cutoff_freqs},
            reject_freq={self.reject_freq},
            apply_mask={self.apply_mask},
            probe_groups={self.probe_groups},
            down_sample_rate={self.down_sample_rate},
            down_sample_after_filter={self.down_sample_after_filter},
        )"""

    def down_sample(self, eeg: Tensor, eeg_mask: Tensor) -> tuple[Tensor, Tensor]:
        eeg = AF.resample(
            eeg, self.sampling_rate, self.sampling_rate // self.down_sample_rate
        )
        eeg_mask = AF.resample(
            eeg_mask,
            self.sampling_rate,
            self.sampling_rate // self.down_sample_rate,
        )
        return eeg, eeg_mask

    @torch.no_grad()
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        return:
        - eeg: (B, C, T)
        - eeg_mask: (B, C, T)
        """
        eegs = []
        eeg_masks = []

        if mask is None:
            mask = torch.ones_like(x)

        for _, probes in self.probe_groups.items():
            for p1, p2 in zip(probes[:-1], probes[1:]):
                x_diff = x[..., PROBE2IDX[p1]] - x[..., PROBE2IDX[p2]]
                if self.apply_mask:
                    x_diff *= mask[..., PROBE2IDX[p1]] * mask[..., PROBE2IDX[p2]]
                eegs.append(x_diff)
                eeg_masks.append(mask[..., PROBE2IDX[p1]] * mask[..., PROBE2IDX[p2]])

        eegs = torch.stack(eegs, dim=1)
        eeg_masks = torch.stack(eeg_masks, dim=1)

        with torch.autocast(device_type="cuda", enabled=False):
            if not self.down_sample_after_filter:
                eegs, eeg_masks = self.down_sample(eegs, eeg_masks)

            if self.reject_freq is not None:
                eegs = AF.bandreject_biquad(eegs, self.sampling_rate, self.reject_freq)
            if self.cutoff_freqs[0] is not None:
                eegs = AF.highpass_biquad(
                    eegs, self.sampling_rate, self.cutoff_freqs[0]
                )
            if self.cutoff_freqs[1] is not None:
                eegs = AF.lowpass_biquad(eegs, self.sampling_rate, self.cutoff_freqs[1])

            if self.down_sample_after_filter:
                eegs, eeg_masks = self.down_sample(eegs, eeg_masks)

        output = dict(eeg=eegs, eeg_mask=eeg_masks)
        return output
