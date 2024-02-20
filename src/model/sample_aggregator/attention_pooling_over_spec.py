import torch.nn as nn
from einops import rearrange
from torch import Tensor

from src.model.basic_block import GatedSpecAttention


class AttentionPoolingOverSpec(nn.Module):
    """
    スペクトログラムのタテヨコの次元に対してAttention Poolingを行う
    """

    def __init__(
        self,
        input_channels: int,
        hidden_size: int = 64,
        temporal_smooth_kernel_size: int = 3,
        frequency_smooth_kernel_size: int = 3,
        attend_to_frequency_dim: bool = True,
    ):
        super().__init__()
        self.channel_mapper = nn.Conv2d(input_channels, hidden_size, kernel_size=1)
        self.att = GatedSpecAttention(
            hidden_size=hidden_size,
            attend_to_frequency_dim=attend_to_frequency_dim,
            temporal_smooth_kernel_size=temporal_smooth_kernel_size,
            frequency_smooth_kernel_size=frequency_smooth_kernel_size,
        )
        self.attend_to_frequency_dim = attend_to_frequency_dim

        self._output_size = hidden_size

    @property
    def output_size(self) -> int:
        return self._output_size

    def forward(self, x: Tensor, num_samples: int) -> Tensor:
        """
        Attention Pooling over training/validation samples

        x: (B, C, F, (T * S))

        return
        ------
        x: (B, C, F, 1) or (B, C, 1, 1)
        """
        x = self.channel_mapper(x)
        x = x * self.att(x)

        if self.attend_to_frequency_dim:
            x = rearrange(x, "b c f t -> b c (f t)")
            x = x.sum(dim=-1, keepdim=True)
            x = x.unsqueeze(-1)
        else:
            x = x.sum(dim=-1, keepdim=True)

        return x
