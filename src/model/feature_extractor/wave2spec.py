import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF
from torchaudio.transforms import MelSpectrogram

from src.constant import PROBE2IDX


class Wave2Spectrogram(nn.Module):
    def __init__(
        self,
        sampling_rate=40,
        n_fft=256,
        win_length=256,
        hop_length=32,
        probes_ll=["Fp1", "F7", "T3", "T5", "O1"],
        probes_rl=["Fp2", "F8", "T4", "T6", "O2"],
        probes_lp=["Fp1", "F3", "C3", "P3", "O1"],
        probes_rp=["Fp2", "F4", "C4", "P4", "O2"],
        probes_z=["Fz", "Cz", "Pz"],
        cutoff_freqs=(0.5, None),
        frequency_lim=(0, 20),
        db_cutoff=60,
        db_offset=0,
        n_mels=128,
        window_fn=torch.hann_window,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.probes_ll = probes_ll
        self.probes_rl = probes_rl
        self.probes_lp = probes_lp
        self.probes_rp = probes_rp
        self.probes_z = probes_z
        self.cutoff_freqs = cutoff_freqs
        self.db_cutoff = db_cutoff
        self.db_offset = db_offset
        self.sampling_rate = sampling_rate

        self.wave2spec = MelSpectrogram(
            sample_rate=sampling_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=frequency_lim[0],
            f_max=frequency_lim[1],
            center=False,
            window_fn=window_fn,
        )

    def same_padding_waveform(self, signals: torch.Tensor, value=0, mode="reflect"):
        """
        STFTの出力系列長が「入力系列長//hop_length」となるようにpaddingする
        """
        pad_size = int((self.n_fft - self.hop_length) / 2)
        signals = F.pad(signals, (pad_size, pad_size, 0, 0), mode=mode, value=value)
        return signals

    def downsample(self, x: torch.Tensor):
        """
        1次元信号をダウンサンプリングする
        """
        x = x[..., self.n_fft // 2 :: self.hop_length]
        return x

    @torch.no_grad
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, threshold=0.5):
        _, num_frames, _ = x.shape
        spectrograms = []
        signals = []
        probe_pairs = []
        probe_groups = []
        channel_masks = []

        if mask is None:
            mask = torch.ones_like(x)

        for probes, probe_group in zip(
            [
                self.probes_ll,
                self.probes_lp,
                self.probes_z,
                self.probes_rp,
                self.probes_rl,
            ],
            ["LL", "LP", "Z", "RP", "RL"],
        ):
            signal = []
            channel_mask = []
            for p1, p2 in zip(probes[:-1], probes[1:]):
                x_diff = x[..., PROBE2IDX[p1]] - x[..., PROBE2IDX[p2]]
                x_diff *= mask[..., PROBE2IDX[p1]] * mask[..., PROBE2IDX[p2]]
                signal.append(x_diff)
                channel_mask.append(mask[..., PROBE2IDX[p1]] * mask[..., PROBE2IDX[p2]])

            signal = torch.stack(signal, dim=1)

            if self.cutoff_freqs[0] is not None:
                signal = AF.highpass_biquad(
                    signal, self.sampling_rate, self.cutoff_freqs[0]
                )
            if self.cutoff_freqs[1] is not None:
                signal = AF.lowpass_biquad(
                    signal, self.sampling_rate, self.cutoff_freqs[1]
                )

            for i, (p1, p2) in enumerate(zip(probes[:-1], probes[1:])):
                signals.append(signal[:, i, :])
                probe_pairs.append((p1, p2))
                probe_groups.append(probe_group)

            channel_mask = torch.stack(channel_mask, dim=1)
            channel_mask_org = channel_mask.clone()
            channel_mask = self.same_padding_waveform(
                channel_mask, mode="constant", value=1
            )
            channel_mask = self.downsample(channel_mask)
            channel_mask = channel_mask.unsqueeze(dim=2)

            spec = self.same_padding_waveform(signal)
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
            channel_mask = channel_mask[..., :T]
            BM, CM, FM, TM = channel_mask.shape
            assert B == BM and C == CM and T == TM, (spec.shape, channel_mask.shape)

            spec = spec * channel_mask
            spec = spec.sum(dim=1) / channel_mask.sum(dim=1)

            spectrograms.append(spec)
            channel_masks.append(channel_mask_org)

        spectrograms = torch.stack(spectrograms, dim=1)
        signals = torch.stack(signals, dim=1)
        channel_masks = torch.concat(channel_masks, dim=1)
        spectrograms = torch.nan_to_num(spectrograms, nan=-self.db_cutoff)

        assert (
            spectrograms.shape[-1] == num_frames // self.hop_length
        ), spectrograms.shape
        output = dict(
            spectrogram=spectrograms,
            signal=signals,
            channel_mask=channel_masks,
            probe_pairs=probe_pairs,
            probe_groups=probe_groups,
        )
        return output
