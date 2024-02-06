from importlib import import_module

import torch
import torch.nn as nn
import torchaudio.functional as AF
from torchaudio.transforms import MelSpectrogram

from src.constant import PROBE2IDX, PROBE_GROUPS
from src.model.tensor_util import rolling_mean, same_padding_1d


class Wave2Spectrogram(nn.Module):
    def __init__(
        self,
        sampling_rate=40,
        n_fft=256,
        win_length=256,
        hop_length=64,
        cutoff_freqs=(0.5, 50),
        frequency_lim=(0, 20),
        db_cutoff=60,
        db_offset=0,
        n_mels=128,
        window_fn="hann_window",
        apply_mask=True,
        downsample_mode="linear",
    ):
        super().__init__()
        torch_module = import_module("torch")

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.cutoff_freqs = cutoff_freqs
        self.db_cutoff = db_cutoff
        self.db_offset = db_offset
        self.sampling_rate = sampling_rate
        self.apply_mask = apply_mask
        self.downsample_mode = downsample_mode

        self.wave2spec = MelSpectrogram(
            sample_rate=sampling_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=frequency_lim[0],
            f_max=frequency_lim[1],
            center=False,
            window_fn=getattr(torch_module, window_fn),
        )

    def downsample_mask(self, x: torch.Tensor, mode="nearest") -> torch.Tensor:
        """
        1次元信号をダウンサンプリングする
        """
        x = same_padding_1d(
            x,
            kernel_size=self.n_fft,
            stride=self.hop_length,
            mode="replicate",
        )
        if mode == "nearest":
            x = x[..., self.n_fft // 2 :: self.hop_length]
        elif mode == "linear":
            x = rolling_mean(
                x,
                kernel_size=self.n_fft,
                stride=self.hop_length,
                apply_padding=False,
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")

        return x

    # @torch.no_grad()
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        return:
        - signal: (B, C, T)
        - spectrogram: (B, C, F, T)
        - channel_mask: (B, C, T)
        - spec_mask: (B, C, 1, T)
        """
        _, num_frames, _ = x.shape
        spectrograms = []
        signals = []
        probe_pairs = []
        probe_groups = []
        channel_masks = []
        spec_masks = []

        if mask is None:
            mask = torch.ones_like(x)

        for probe_group, probes in PROBE_GROUPS.items():
            signal = []
            channel_mask = []
            for p1, p2 in zip(probes[:-1], probes[1:]):
                x_diff = x[..., PROBE2IDX[p1]] - x[..., PROBE2IDX[p2]]
                if self.apply_mask:
                    x_diff *= mask[..., PROBE2IDX[p1]] * mask[..., PROBE2IDX[p2]]
                signal.append(x_diff)
                channel_mask.append(mask[..., PROBE2IDX[p1]] * mask[..., PROBE2IDX[p2]])

            signal = torch.stack(signal, dim=1)
            channel_mask = torch.stack(channel_mask, dim=1)

            if self.cutoff_freqs[0] is not None:
                signal = AF.highpass_biquad(
                    signal, self.sampling_rate, self.cutoff_freqs[0]
                )
            if self.cutoff_freqs[1] is not None:
                signal = AF.lowpass_biquad(
                    signal, self.sampling_rate, self.cutoff_freqs[1]
                )

            spec_mask = self.downsample_mask(channel_mask, mode=self.downsample_mode)
            spec_mask = spec_mask.unsqueeze(dim=2)

            spec = same_padding_1d(
                signal, kernel_size=self.n_fft, stride=self.hop_length, mode="reflect"
            )
            spec = self.wave2spec(spec)
            spec = (
                AF.amplitude_to_DB(
                    spec,
                    multiplier=10.0,
                    amin=1e-8,
                    top_db=self.db_cutoff,
                    db_multiplier=0,
                )
                + self.db_offset
            )
            B, C, F, T = spec.shape
            spec_mask = spec_mask[..., :T]
            BM, CM, FM, TM = spec_mask.shape
            assert B == BM and C == CM and T == TM, (spec.shape, spec_mask.shape)

            # if apply_mask:
            #    spec = spec * spec_mask
            #    spec = spec.sum(dim=1) / spec_mask.sum(dim=1)
            # else:
            #    spec = spec.mean(dim=1)

            for i, (p1, p2) in enumerate(zip(probes[:-1], probes[1:])):
                signals.append(signal[:, i, :])
                channel_masks.append(channel_mask[:, i, :])
                spectrograms.append(spec[:, i, :])
                spec_masks.append(spec_mask[:, i, :])
                probe_pairs.append((p1, p2))
                probe_groups.append(probe_group)

        spectrograms = torch.stack(spectrograms, dim=1)
        signals = torch.stack(signals, dim=1)
        channel_masks = torch.stack(channel_masks, dim=1)
        spec_masks = torch.stack(spec_masks, dim=1)
        spectrograms = torch.nan_to_num(spectrograms, nan=-self.db_cutoff)

        assert (
            spectrograms.shape[-1] == num_frames // self.hop_length
        ), spectrograms.shape
        output = dict(
            spectrogram=spectrograms,
            signal=signals,
            channel_mask=channel_masks,
            spec_mask=spec_masks,
            probe_pairs=probe_pairs,
            probe_groups=probe_groups,
        )
        return output
