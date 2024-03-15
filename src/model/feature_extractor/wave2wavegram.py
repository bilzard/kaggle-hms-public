import torch
import torch.nn as nn
from einops import rearrange

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
        eeg = output["eeg"]
        eeg_mask = output["eeg_mask"]

        if mask is None:
            mask = torch.ones_like(x)

        spec_mask = self.downsample_mask(eeg_mask, mode=self.downsample_mode)
        spec_mask = spec_mask.unsqueeze(dim=2)

        B, Ch, T = eeg.shape
        eeg = rearrange(eeg, "b ch t -> (b ch) 1 t")
        spec = self.wavegram(eeg)
        spec = rearrange(spec, "(b ch) 1 f t -> b ch f t", b=B, ch=Ch)
        B, C, F, T = spec.shape
        spec_mask = spec_mask[..., :T]
        BM, CM, FM, TM = spec_mask.shape
        assert B == BM and C == CM and T == TM, (spec.shape, spec_mask.shape)

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

    eeg = torch.randn(batch_size, num_frames, num_probes)
    mask = torch.rand(batch_size, num_frames, num_probes)
    wavegram = Wavegram(
        1,
        1,
        hidden_dims=hidden_dims,
        num_filter_banks=num_filter_banks,
    )

    model = Wave2Wavegram(
        wavegram,
        win_length=win_length,
    )
    output = model(eeg, mask)
    assert output["spec"].shape == (
        batch_size,
        num_channels,
        num_filter_banks,
        num_frames // 2 ** (len(hidden_dims) - 1),
    ), output["spec"].shape

    summary(model, input_size=(batch_size, num_frames, num_probes))
