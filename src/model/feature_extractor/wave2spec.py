import torch
import torch.nn as nn
from torch import Tensor

from src.model.feature_extractor.eeg import ChannelCollator
from src.model.tensor_util import rolling_mean, same_padding_1d


class Wave2Spectrogram(nn.Module):
    def __init__(
        self,
        wave2spec: nn.Module,
        cutoff_freqs: tuple[float, float] = (0.5, 50),
        apply_mask: bool = True,
        downsample_mode: str = "linear",
        expand_mask: bool = True,
    ):
        super().__init__()

        self.cutoff_freqs = cutoff_freqs
        self.apply_mask = apply_mask
        self.downsample_mode = downsample_mode
        self.expand_mask = expand_mask

        self.win_length = wave2spec.win_length
        self.hop_length = wave2spec.hop_length

        self.collate_channels = ChannelCollator(
            sampling_rate=wave2spec.sampling_rate,
            cutoff_freqs=cutoff_freqs,
            apply_mask=apply_mask,
        )
        self.wave2spec = wave2spec

    @torch.no_grad()
    def downsample_mask(self, x: torch.Tensor, mode="nearest") -> torch.Tensor:
        """
        1次元信号をダウンサンプリングする
        """
        x = same_padding_1d(
            x,
            kernel_size=self.win_length,
            stride=self.hop_length,
            mode="replicate",
        )
        if mode == "nearest":
            x = x[..., self.win_length // 2 :: self.hop_length]
        elif mode == "linear":
            x = rolling_mean(
                x,
                kernel_size=self.win_length,
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
        return self.forward_spec(output)

    def forward_spec(self, output: dict[str, Tensor]):
        eeg = output["eeg"]
        eeg_mask = output["eeg_mask"]

        _, _, num_frames = eeg.shape

        spec_mask = self.downsample_mask(eeg_mask, mode=self.downsample_mode)
        spec_mask = spec_mask.unsqueeze(dim=2)

        spec = self.wave2spec(eeg)
        B, C, F, T = spec.shape
        spec_mask = spec_mask[..., :T]
        BM, CM, FM, TM = spec_mask.shape
        assert B == BM and C == CM and T == TM, (spec.shape, spec_mask.shape)

        if self.expand_mask:
            F = spec.shape[2]
            spec_mask = spec_mask.expand(-1, -1, F, -1)

        assert (
            spec.shape[-1] == num_frames // self.hop_length
        ), f"{spec.shape=}, {num_frames=}, {self.hop_length=}"
        output = dict(spec=spec, eeg=eeg, eeg_mask=eeg_mask, spec_mask=spec_mask)
        return output


if __name__ == "__main__":
    from torchinfo import summary

    from src.model.feature_extractor.module import MelSpec

    batch_size = 2
    num_probes = 19
    num_channels = num_probes - 1
    num_frames = 2048
    sampling_rate = 40
    n_fft = 256
    win_length = 64
    hop_length = 16
    eeg = torch.randn(batch_size, num_frames, num_probes)
    mask = torch.randn(batch_size, num_frames, num_probes)
    wave2spec = MelSpec(n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    print(wave2spec)
    model = Wave2Spectrogram(wave2spec)
    output = model(eeg, mask)
    assert output["spec"].shape == (
        batch_size,
        num_channels,
        n_fft // 2,
        num_frames // hop_length,
    )
    assert output["spec_mask"].shape == (
        batch_size,
        num_channels,
        n_fft // 2,
        num_frames // hop_length,
    ), f"{output['spec_mask'].shape=}"
    summary(model, input_size=(batch_size, num_frames, num_probes))
