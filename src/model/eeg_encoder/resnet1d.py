"""
ResNet 1d architecture.

Refactor and modified by @bilzard.

Reference
---------
[1] https://www.kaggle.com/code/nischaydnk/lightning-1d-eegnet-training-pipeline-hbs?scriptVersionId=160814948
"""

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from src.model.basic_block import GruBlock


class SeBlock(nn.Module):
    def __init__(self, hidden_dim: int, se_ratio: int = 4):
        super().__init__()

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Conv1d(hidden_dim, hidden_dim // se_ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // se_ratio, hidden_dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class ConvBnRelu1d(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
    ):
        padding = (kernel_size - stride) // 2 if padding is None else padding
        super().__init__(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )


class ResBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 2,
        se_ratio: int = 4,
    ):
        super().__init__()
        self.stem_conv = nn.Sequential(
            ConvBnRelu1d(in_channels, out_channels, kernel_size),
            ConvBnRelu1d(out_channels, out_channels),
            SeBlock(out_channels, se_ratio),
            nn.MaxPool1d(kernel_size=stride, stride=stride),
        )
        self.pool = nn.MaxPool1d(kernel_size=stride, stride=stride)

    def forward(self, x: Tensor) -> Tensor:
        return self.stem_conv(x) + self.pool(x)


class ResNet1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        sep_kernel_sizes: tuple[int, ...] = (3, 5, 7, 9),
        kernel_size: int = 3,
        hidden_dim: int = 64,
        se_ratio: int = 2,
        stem_conv_kernel_size: int = 4,
        stem_conv_stride: int = 4,
        res_block_size: int = 6,
        stride_per_block: int = 1,
        num_gru_layers: int = 1,
    ):
        """
        output stride := `2 ** (res_block_size + 1) / len(sep_kernels_sizes)`
        """
        super().__init__()
        self.sep_kernel_sizes = sep_kernel_sizes
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        self.num_gru_layers = num_gru_layers
        self.num_sep_kernels = len(sep_kernel_sizes)

        self.sep_convs = nn.ModuleList(
            ConvBnRelu1d(in_channels, hidden_dim, k) for k in sep_kernel_sizes
        )
        self.stem_conv = ConvBnRelu1d(
            hidden_dim,
            hidden_dim,
            kernel_size=stem_conv_kernel_size,
            stride=stem_conv_stride,
        )
        self.resnet = nn.Sequential(
            *[
                ResBlock1d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    stride=stride_per_block,
                    se_ratio=se_ratio,
                )
                for _ in range(res_block_size)
            ],
        )
        self.mapper = nn.Sequential(
            ConvBnRelu1d(hidden_dim * self.num_sep_kernels, hidden_dim),
        )
        self.gru = nn.Sequential(
            *[
                GruBlock(hidden_dim, n_layers=1, bidirectional=True)
                for _ in range(num_gru_layers)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, C, T)
        output: (B, C, T)
        """
        # NOTE: 元のコードにしたがってT方向にtilingしているが、batch方向のstackingでも同様の結果が得られるかもしれない.
        # 他には、channel方向に結合してearly fusionにする選択肢がある.
        x = torch.cat([sep_conv(x) for sep_conv in self.sep_convs], dim=2)
        x = self.stem_conv(x)
        x = self.resnet(x)
        x = rearrange(x, "b c (s t) -> b (s c) t", s=len(self.sep_kernel_sizes))
        x = self.mapper(x)

        if self.num_gru_layers > 0:
            x = rearrange(x, "b c t -> b t c")
            x = self.gru(x)
            x = rearrange(x, "b t c -> b c t")

        return x


if __name__ == "__main__":
    from torchinfo import summary

    in_channels = 2
    num_frames = 2048

    model = ResNet1d(in_channels)
    summary(model, input_size=(2, in_channels, num_frames))
