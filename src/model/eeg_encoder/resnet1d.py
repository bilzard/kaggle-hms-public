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
        assert (
            kernel_size >= stride
        ), f"kernel_size must be greater than stride. Got {kernel_size} and {stride}."
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


class ParallelConv(nn.Module):
    """
    Parallel conv + upsample
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernels: list[int],
    ):
        super().__init__()
        self.sep_convs = nn.ModuleList(
            [ConvBnRelu1d(in_channels, out_channels, kernel_size=k) for k in kernels]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, C, T)
        output: (B, C, T)
        """
        feats = []
        for sep_conv in self.sep_convs:
            feats.append(sep_conv(x))

        x = torch.cat(feats, dim=2)
        return x


class ResNet1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        sep_kernels: list[int] = [3, 5, 7, 9],
        hidden_dim: int = 64,
        se_ratio: int = 2,
        num_res_blocks: int = 6,
        kernel_size: int = 3,
        stride: int = 2,
        stem_kernel_size: int = 4,
        stem_stride: int = 2,
    ):
        """
        output stride := `2 ** (res_block_size + 1) / len(sep_kernels_sizes)`
        """
        super().__init__()
        self.in_channels = in_channels
        self.sep_kernels = sep_kernels
        self.hidden_dim = hidden_dim
        self.num_res_blocks = num_res_blocks

        self.parallel_conv = ParallelConv(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernels=sep_kernels,
        )
        self.stem = ConvBnRelu1d(hidden_dim, hidden_dim, stem_kernel_size, stem_stride)
        self.resnet = nn.Sequential(
            *[
                ResBlock1d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    se_ratio=se_ratio,
                )
                for _ in range(num_res_blocks)
            ],
        )
        self.mapper = nn.Sequential(
            ConvBnRelu1d(hidden_dim * len(self.sep_kernels), hidden_dim),
        )

    @property
    def out_channels(self) -> int:
        return self.hidden_dim

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, C, T)
        output: (B, C, T)
        """
        x = self.parallel_conv(x)  # 1/1
        x = self.stem(x)
        x = self.resnet(x)  # 1/128
        x = rearrange(x, "b c (s t) -> b (s c) t", s=len(self.sep_kernels))
        x = self.mapper(x)

        return x


if __name__ == "__main__":
    from torchinfo import summary

    in_channels = 2
    num_frames = 2048

    model = ResNet1d(in_channels)
    summary(model, input_size=(2, in_channels, num_frames))
