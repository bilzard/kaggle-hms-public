from importlib import import_module

import torch
import torch.nn as nn
import torchaudio.functional as AF
from torchaudio.transforms import MelSpectrogram

from src.model.tensor_util import same_padding_1d


class MelSpec(nn.Module):
    def __init__(
        self,
        sampling_rate: int = 40,
        n_fft: int = 256,
        win_length: int = 64,
        hop_length: int = 16,
        frequency_lim: tuple[int, int] = (0, 20),
        db_cutoff: int = 60,
        db_offset: int = 0,
        n_mels: int = 128,
        window_fn: str = "hann_window",
        norm: tuple[float, float] = (-37.52, 16.10),
    ):
        super().__init__()
        torch_module = import_module("torch")
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.db_cutoff = db_cutoff
        self.db_offset = db_offset
        self.frequency_lim = frequency_lim
        self.n_mels = n_mels
        self.window_fn = window_fn
        self.norm = norm

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
            frequency_lim={self.frequency_lim},
            db_cutoff={self.db_cutoff},
            db_offset={self.db_offset},
            n_mels={self.n_mels},
            window_fn={self.window_fn},
        )"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: b c t
        return: b c f t
        """
        x = same_padding_1d(
            x, kernel_size=self.n_fft, stride=self.hop_length, mode="reflect"
        )
        with torch.autocast(device_type="cuda", enabled=False):
            x = self.wave2spec(x)
            x = (
                AF.amplitude_to_DB(
                    x,
                    multiplier=10.0,
                    amin=1e-8,
                    top_db=self.db_cutoff,
                    db_multiplier=0,
                )
                + self.db_offset
            )
            x = (x - self.norm[0]) / self.norm[1]
        return x


if __name__ == "__main__":
    from torchinfo import summary

    batch_size = 2
    num_frames = 2048
    in_channels = 19

    input = torch.randn(batch_size, in_channels, num_frames)
    model = MelSpec()
    output = model(input)

    summary(model, input_size=(batch_size, in_channels, num_frames))
