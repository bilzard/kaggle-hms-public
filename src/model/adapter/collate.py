import torch.nn as nn
from einops import rearrange
from torch import Tensor


class WavegramCollator(nn.Module):
    """
    Wavegramの複数チャネルに対応したcollator.
    """

    def __init__(self, num_planes):
        super().__init__()
        self.num_planes = num_planes

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_planes={self.num_planes})"

    def forward(self, spec: Tensor, spec_mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        eeg: (b p) c f t
        eeg_mask: (b p) c f t

        p: EEG channelあたりの次元
        """
        spec = rearrange(spec, "(b p) c f t -> b (c p) f t", p=self.num_planes)
        spec_mask = rearrange(
            spec_mask, "(b p) c f t -> b (c p) f t", p=self.num_planes
        )

        return spec, spec_mask
