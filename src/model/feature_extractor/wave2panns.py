import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torchaudio.transforms import FrequencyMasking, TimeMasking

from src.model.feature_extractor import ChannelCollator
from src.model.tensor_util import rolling_mean, same_padding_1d


class Wave2Panns(nn.Module):
    """
    Wavegram[1]によるfilter bankの抽出

    Reference:
    ----------
    [1] PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition
    """

    def __init__(
        self,
        channel_collator: nn.Module,
        wavegram: nn.Module,
        wave2spec: nn.Module,
        cutoff_freqs: tuple[float, float] = (0.5, 50),
        apply_mask: bool = True,
        downsample_mode: str = "linear",
        expand_mask: bool = True,
        freq_mask_param: int = 16,
        time_mask_param: int = 32,
    ):
        super().__init__()

        self.cutoff_freqs = cutoff_freqs
        self.apply_mask = apply_mask
        self.downsample_mode = downsample_mode
        self.expand_mask = expand_mask

        self.win_length = wave2spec.win_length
        self.hop_length = wave2spec.hop_length

        self.collate_channels = channel_collator
        self.wavegram = wavegram
        self.wave2spec = wave2spec

        self.spec_aug = torch.nn.Sequential(
            FrequencyMasking(freq_mask_param),
            TimeMasking(time_mask_param),
        )

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

        spec_mask = self.downsample_mask(eeg_mask, mode=self.downsample_mode)
        spec_mask = spec_mask.unsqueeze(dim=2)

        spec = self.wave2spec(eeg)

        B, Ch, T = eeg.shape
        wavegram = rearrange(eeg, "b ch t -> (b ch) 1 t")
        wavegram = self.wavegram(wavegram)
        wavegram = rearrange(wavegram, "(b ch) 1 f t -> b ch f t", b=B, ch=Ch)

        spec = torch.cat([spec, wavegram], dim=2)

        if self.training:
            spec = rearrange(spec, "b ch (m f) t -> b (ch m) f t", m=2)
            spec = self.spec_aug(spec)
            spec = rearrange(spec, "b (ch m) f t -> b ch (m f) t", m=2)

        # expand mask
        B, C, _, T = spec.shape
        spec_mask = spec_mask[..., :T]
        BM, CM, _, TM = spec_mask.shape
        assert B == BM and C == CM and T == TM, (spec.shape, spec_mask.shape)
        if self.expand_mask:
            spec_mask = spec_mask.expand(-1, -1, spec.shape[2], -1)

        output = dict(
            spec=spec,
            eeg=eeg,
            eeg_mask=eeg_mask,
            spec_mask=spec_mask,
        )
        return output


if __name__ == "__main__":
    from torchinfo import summary

    from src.model.feature_extractor.module import MelSpec, Wavegram

    batch_size = 2
    num_frames = 10240
    num_probes = 19
    num_channels = num_probes - 1
    num_filter_banks = 64
    hidden_dims = [64, 64, 64, 128, 128]
    original_sampling_rate = 200
    sampling_rate = 40
    down_sample_rate = 5
    n_fft = 256
    win_length = 64
    hop_length = 16
    n_mels = 64

    eeg = torch.randn(batch_size, num_frames, num_probes)
    mask = torch.rand(batch_size, num_frames, num_probes)
    wavegram = Wavegram(
        1,
        1,
        hidden_dims=hidden_dims,
        num_filter_banks=num_filter_banks,
    )
    wave2spec = MelSpec(
        n_fft=n_fft, win_length=win_length, hop_length=hop_length, n_mels=n_mels
    )
    channel_collator = ChannelCollator(
        sampling_rate=original_sampling_rate,
        cutoff_freqs=(0.5, 50),
        apply_mask=True,
        down_sample_rate=down_sample_rate,
    )
    print(wave2spec)

    model = Wave2Panns(channel_collator, wavegram, wave2spec)
    output = model(eeg, mask)
    assert output["spec"].shape == (
        batch_size,
        num_channels,
        n_mels + num_filter_banks,
        num_frames // down_sample_rate // hop_length,
    ), f"{output['spec'].shape=}"
    assert output["spec_mask"].shape == (
        batch_size,
        num_channels,
        n_mels + num_filter_banks,
        num_frames // down_sample_rate // hop_length,
    ), f"{output['spec_mask'].shape=}"
    print("spec", output["spec"].shape)
    print("spec_mask", output["spec_mask"].shape)

    summary(model, input_size=(batch_size, num_frames, num_probes))
