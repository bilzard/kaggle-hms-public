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


class ConvBnSiLu1d(nn.Sequential):
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
            nn.SiLU(inplace=True),
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
            ConvBnSiLu1d(in_channels, out_channels, kernel_size, stride=stride),
            ConvBnSiLu1d(out_channels, out_channels),
            SeBlock(out_channels, se_ratio),
        )
        self.skip = nn.Sequential(
            nn.MaxPool1d(
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - stride) // 2,
            ),
            ConvBnSiLu1d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.stem_conv(x) + self.skip(x)


class ParallelConv(nn.Module):
    """
    Parallel conv + upsample
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernels: list[int],
        strides: list[int],
        scale_factor: int,
    ):
        super().__init__()
        self.sep_convs = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBnSiLu1d(in_channels, out_channels, kernel_size=k, stride=s),
                    nn.Upsample(scale_factor=s // scale_factor),
                )
                for k, s in zip(kernels, strides)
            ]
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


class ResNet1dV2(nn.Module):
    def __init__(
        self,
        in_channels: int,
        sep_kernels: list[int] = [6, 12, 24, 48],
        sep_strides: list[int] = [4, 8, 16, 32],
        sep_scale_factor: int = 1,
        sep_hidden_dim: int = 32,
        res_kernel_sizes: list[int] = [12, 6, 6],
        res_strides: list[int] = [8, 4, 4],
        res_hidden_dims: list[int] = [32, 16, 24, 40],
        se_ratio: int = 4,
        aggregate_filters: bool = True,
    ):
        """
        output stride := `2 ** (res_block_size + 1) / len(sep_kernels_sizes)`
        """
        super().__init__()
        self.in_channels = in_channels
        self.sep_kernels = sep_kernels
        self.sep_strides = sep_strides
        self.res_kernel_sizes = res_kernel_sizes
        self.res_strides = res_strides
        self.res_hidden_dims = res_hidden_dims
        self.sep_scale_factor = sep_scale_factor
        self.aggregate_filters = aggregate_filters

        self.parallel_conv = ParallelConv(
            in_channels=in_channels,
            out_channels=sep_hidden_dim,
            kernels=sep_kernels,
            strides=sep_strides,
            scale_factor=sep_scale_factor,
        )
        self.resnet = nn.Sequential(
            *[
                ResBlock1d(
                    in_channels=di,
                    out_channels=do,
                    kernel_size=k,
                    stride=s,
                    se_ratio=se_ratio,
                )
                for k, s, di, do in zip(
                    res_kernel_sizes,
                    res_strides,
                    res_hidden_dims[:-1],
                    res_hidden_dims[1:],
                )
            ],
        )

    @property
    def out_channels(self) -> int:
        return (
            self.res_hidden_dims[-1] * len(self.sep_kernels)
            if self.aggregate_filters
            else self.res_hidden_dims[-1]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, C, T)
        output: (B, C, T)
        """
        x = self.parallel_conv(x)  # 1/1
        x = self.resnet(x)  # 1/128
        if self.aggregate_filters:
            x = rearrange(x, "b c (s t) -> b (s c) t", s=len(self.sep_kernels))  # 1/4

        return x


if __name__ == "__main__":
    from torchinfo import summary

    in_channels = 2
    num_frames = 2048

    model = ResNet1dV2(in_channels)
    summary(model, input_size=(2, in_channels, num_frames))
