import torch
import torch.nn as nn
import torchaudio.functional as AF

from src.constant import PROBE2IDX, PROBE_GROUPS


class ChannelCollator(nn.Module):
    def __init__(
        self,
        sampling_rate: int = 40,
        cutoff_freqs: tuple[float, float] = (0.5, 50),
        apply_mask: bool = True,
        probe_groups: dict[str, list[str]] = PROBE_GROUPS,
        clip_val: float | None = None,
    ):
        super().__init__()

        self.sampling_rate = sampling_rate
        self.cutoff_freqs = cutoff_freqs
        self.apply_mask = apply_mask
        self.probe_groups = probe_groups
        self.clip_val = clip_val

    def __repr__(self):
        return f"""{self.__class__.__name__}(
            sampling_rate={self.sampling_rate},
            cutoff_freqs={self.cutoff_freqs},
            apply_mask={self.apply_mask},
            probe_groups={self.probe_groups},
            clip_val={self.clip_val},
        )"""

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
            if self.cutoff_freqs[0] is not None:
                eegs = AF.highpass_biquad(
                    eegs, self.sampling_rate, self.cutoff_freqs[0]
                )
            if self.cutoff_freqs[1] is not None:
                eegs = AF.lowpass_biquad(eegs, self.sampling_rate, self.cutoff_freqs[1])

        if self.clip_val is not None:
            eegs = eegs.clamp(-self.clip_val, self.clip_val)

        output = dict(eeg=eegs, eeg_mask=eeg_masks)
        return output
