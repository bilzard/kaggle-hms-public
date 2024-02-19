"""
Gated attention modules

The original idea of gated attention is proposed in the paper[1].
The author's original implementation in Keras is available in [2].

Reference
---------
[1] Attention-based Deep Multiple Instance Learning, https://arxiv.org/abs/1802.04712
[2] https://github.com/utayao/Atten_Deep_MIL/blob/master/utl/custom_layers.py
"""

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor


class GatedMilAttention(nn.Module):
    """
    Calculate attention score over multiple instance dimension.
    """

    def __init__(self, hidden_size: int):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.Sigmoid(),
        )
        self.prop = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.Tanh(),
        )
        self.map_score = nn.Conv1d(hidden_size, 1, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, C, I)
            - B: the number of batches
            - C: the number of channels
            - I: the number of instances

        output
        ------
        att_score: (B, C, I)
        """
        x = self.map_score(self.prop(x) * self.gate(x))
        x = torch.softmax(x, dim=-1)

        return x


class GatedSpecAttention(nn.Module):
    """
    Calculate attention score over temporal (and frequency) dimension.
    """

    def __init__(
        self,
        hidden_size: int,
        attend_to_frequency_dim: bool = True,
        frequency_smooth_kernel_size: int = 3,
        temporal_smooth_kernel_size: int = 3,
    ):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.Sigmoid(),
        )
        self.prop = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.Tanh(),
        )
        self.map_score = nn.Conv2d(
            hidden_size,
            1,
            kernel_size=(frequency_smooth_kernel_size, temporal_smooth_kernel_size),
            padding=(
                frequency_smooth_kernel_size // 2,
                temporal_smooth_kernel_size // 2,
            ),
        )
        self.attend_to_frequency_dim = attend_to_frequency_dim

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, C, F, T)

        output
        ------
        att_score: (B, C, F, T)
        """
        x = self.map_score(self.prop(x) * self.gate(x))

        if self.attend_to_frequency_dim:
            B, C, F, T = x.shape
            x = rearrange(x, "b c f t -> b c (f t)")
            x = torch.softmax(x, dim=-1)
            x = rearrange(x, "b c (f t) -> b c f t", f=F, t=T)
        else:
            # only attend to temporal dimension
            x = torch.mean(x, dim=2, keepdim=True)  # pool frequency dimension
            x = torch.softmax(x, dim=-1)

        return x
