from importlib import import_module

import torch
import torch.nn as nn
import torchaudio.functional as AF
from torchaudio.transforms import MelSpectrogram

from src.model.tensor_util import rolling_mean, same_padding_1d

from .eeg import ChannelCollator


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
        expand_mask=True,
    ):
        super().__init__()
        torch_module = import_module("torch")

        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.cutoff_freqs = cutoff_freqs
        self.frequency_lim = frequency_lim
        self.db_cutoff = db_cutoff
        self.db_offset = db_offset
        self.n_mels = n_mels
        self.window_fn = window_fn
        self.apply_mask = apply_mask
        self.downsample_mode = downsample_mode
        self.expand_mask = expand_mask

        self.collate_channels = ChannelCollator(
            sampling_rate=sampling_rate,
            cutoff_freqs=cutoff_freqs,
            apply_mask=apply_mask,
        )
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

    def __repr__(self):
        return f"""{self.__class__.__name__}(
            sampling_rate={self.sampling_rate},
            n_fft={self.n_fft},
            win_length={self.win_length},
            hop_length={self.hop_length},
            cutoff_freqs={self.cutoff_freqs},
            frequency_lim={self.frequency_lim},
            db_cutoff={self.db_cutoff},
            db_offset={self.db_offset},
            n_mels={self.n_mels},
            window_fn={self.window_fn},
            apply_mask={self.apply_mask},
            downsample_mode={self.downsample_mode},
            expand_mask={self.expand_mask},
        )"""

    @torch.no_grad()
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

    @torch.no_grad()
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        return:
        - signal: (B, C, T)
        - spectrogram: (B, C, F, T)
        - channel_mask: (B, C, T)
        - spec_mask: (B, C, 1, T)
        """
        output = self.collate_channels(x, mask)
        eeg = output["eeg"]
        eeg_mask = output["eeg_mask"]

        _, num_frames, _ = x.shape
        if mask is None:
            mask = torch.ones_like(x)

        spec_mask = self.downsample_mask(eeg_mask, mode=self.downsample_mode)
        spec_mask = spec_mask.unsqueeze(dim=2)

        spec = same_padding_1d(
            eeg, kernel_size=self.n_fft, stride=self.hop_length, mode="reflect"
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

        if self.expand_mask:
            F = spec.shape[2]
            spec_mask = spec_mask.expand(-1, -1, F, -1)

        assert spec.shape[-1] == num_frames // self.hop_length, spec.shape
        output = dict(spec=spec, eeg=eeg, eeg_mask=eeg_mask, spec_mask=spec_mask)
        return output
