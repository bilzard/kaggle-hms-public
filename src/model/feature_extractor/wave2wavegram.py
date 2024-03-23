import torch
import torch.nn as nn
from einops import rearrange
from torchaudio.transforms import FrequencyMasking, TimeMasking

from src.model.feature_extractor import ChannelCollator
from src.model.tensor_util import rolling_mean, same_padding_1d


class Wave2Wavegram(nn.Module):
    """
    Wavegram[1]によるfilter bankの抽出

    Reference:
    ----------
    [1] PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition
    """

    def __init__(
        self,
        wavegram: nn.Module,
        sampling_rate: int = 40,
        win_length: int = 64,
        cutoff_freqs: tuple[float, float] = (0.5, 50),
        apply_mask: bool = True,
        downsample_mode: str = "linear",
        expand_mask: bool = True,
        freq_mask_param: int = 16,
        time_mask_param: int = 32,
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.win_length = win_length
        self.cutoff_freqs = cutoff_freqs
        self.apply_mask = apply_mask
        self.downsample_mode = downsample_mode
        self.expand_mask = expand_mask
        self.hop_length = wavegram.hop_length

        self.collate_channels = ChannelCollator(
            sampling_rate=sampling_rate,
            cutoff_freqs=cutoff_freqs,
            apply_mask=apply_mask,
        )
        self.wavegram = wavegram
        self.spec_aug = torch.nn.Sequential(
            FrequencyMasking(freq_mask_param),
            TimeMasking(time_mask_param),
        )

        if wavegram.out_channels > 1:
            self.mask_encoder = nn.Conv2d(
                in_channels=1, out_channels=wavegram.out_channels, kernel_size=1
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
        - signal: b ch t
        - spectrogram: (b c) ch f t
        - channel_mask: b ch t
        - spec_mask: (b c) ch f t
        """
        output = self.collate_channels(x, mask)
        eeg = output["eeg"]
        eeg_mask = output["eeg_mask"]

        if mask is None:
            mask = torch.ones_like(x)

        spec_mask = self.downsample_mask(eeg_mask, mode=self.downsample_mode)
        spec_mask = spec_mask.unsqueeze(dim=2)

        B, Ch, T = eeg.shape
        eeg = rearrange(eeg, "b ch t -> (b ch) 1 t")
        spec = self.wavegram(eeg)  # (b ch) c f t

        if self.training:
            spec = rearrange(spec, "(b ch) c f t -> b (ch c) f t", b=B, ch=Ch)
            spec = self.spec_aug(spec)
            spec = rearrange(spec, "b (ch c) f t -> (b ch) c f t", ch=Ch)

        spec = rearrange(spec, "(b ch) c f t -> (b c) ch f t", b=B, ch=Ch)
        spec_mask = spec_mask[..., :T]
        spec_mask = rearrange(spec_mask, "b ch f t -> (b ch) 1 f t")

        if self.wavegram.out_channels > 1:
            spec_mask = self.mask_encoder(spec_mask)  # (b ch) c f t

        spec_mask = rearrange(spec_mask, "(b ch) c f t -> (b c) ch f t", b=B, ch=Ch)

        assert all([spec.shape[i] == spec_mask.shape[i] for i in [0, 1, 3]]), (
            spec.shape,
            spec_mask.shape,
        )

        if self.expand_mask:
            F = spec.shape[2]
            spec_mask = spec_mask.expand(-1, -1, F, -1)

        output = dict(spec=spec, eeg=eeg, eeg_mask=eeg_mask, spec_mask=spec_mask)
        return output


if __name__ == "__main__":
    from torchinfo import summary

    from src.model.feature_extractor.module import Wavegram

    batch_size = 2
    num_frames = 2048
    num_probes = 19
    num_channels = num_probes - 1
    num_filter_banks = 64
    hidden_dims = [64, 64, 64, 128, 128]
    win_length = 64

    for out_channels in [1, 8]:
        eeg = torch.randn(batch_size, num_frames, num_probes)
        mask = torch.rand(batch_size, num_frames, num_probes)
        wavegram = Wavegram(
            1,
            out_channels,
            hidden_dims=hidden_dims,
            num_filter_banks=num_filter_banks,
        )

        model = Wave2Wavegram(
            wavegram,
            win_length=win_length,
        )
        output = model(eeg, mask)
        assert output["spec"].shape == (
            batch_size * out_channels,
            num_channels,
            num_filter_banks,
            num_frames // 2 ** (len(hidden_dims) - 1),
        ), output["spec"].shape

    summary(model, input_size=(batch_size, num_frames, num_probes))
