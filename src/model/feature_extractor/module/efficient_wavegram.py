"""
efficient wavegram by Depth-wise separative conv/InvertedResidual
"""

import operator
from functools import reduce

import torch
import torch.nn as nn
from einops import rearrange
from timm.layers import DropPath
from torch import Tensor


class SqueezeExcite(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        se_ratio: int,
        activation: type[nn.Module] = nn.SiLU,
    ):
        super().__init__()

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(hidden_dim, hidden_dim // se_ratio, kernel_size=1),
            activation(inplace=True),
            nn.Conv2d(hidden_dim // se_ratio, hidden_dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.se(x)


class ConvBnAct2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, ...] = (1, 1),
        stride: tuple[int, ...] = (1, 1),
        dilation: tuple[int, ...] = (1, 1),
        padding: tuple[int, ...] | str = "same",
        groups: int = 1,
        activation: type[nn.Module] = nn.SiLU,
        drop_path_rate: float = 0.0,
        skip: bool = False,
    ):
        super().__init__()

        match padding:
            case "valid":
                padding = (0, 0)
            case "same":
                valid_kernel_size = tuple(
                    [((k - 1) // 2 * d) * 2 + 1 for k, d in zip(kernel_size, dilation)]
                )
                padding = tuple(
                    [(k - s) // 2 for k, s in zip(valid_kernel_size, stride)]
                )
            case _:
                pass

        self.has_skip = skip and stride == 1 and in_channels == out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,  # type: ignore
                stride,  # type: ignore
                padding,  # type: ignore
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            activation(inplace=True) if activation is not None else nn.Identity(),
        )
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.conv(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut

        return x


class DepthWiseSeparableConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, ...],
        activation: type[nn.Module],
        stride: tuple[int, ...] = (1, 1),
        dilation: tuple[int, ...] = (1, 1),
        se_ratio: int = 4,
        skip: bool = True,
        drop_path_rate: float = 0.0,
        se_after_dw_conv: bool = False,
    ):
        super().__init__()
        self.has_skip = (
            skip and all([s == 1 for s in stride]) and in_channels == out_channels
        )
        modules: list[nn.Module] = [
            ConvBnAct2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                stride=stride,
                groups=in_channels,
                activation=activation,
            )
        ]
        if se_after_dw_conv:
            modules.append(
                SqueezeExcite(in_channels, se_ratio=se_ratio, activation=activation)
            )
        modules.append(
            ConvBnAct2d(
                in_channels,
                out_channels,
                activation=activation,
            )
        )
        if not se_after_dw_conv:
            modules.append(
                SqueezeExcite(out_channels, se_ratio=se_ratio, activation=activation)
            )

        self.conv = nn.Sequential(*modules)
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.conv(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class InvertedResidual(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, ...],
        dilation: tuple[int, ...] = (1, 1),
        stride: tuple[int, ...] = (1, 1),
        depth_multiplier: int = 4,
        se_ratio: int = 16,
        skip: bool = True,
        drop_path_rate: float = 0.0,
        se_after_dw_conv: bool = False,
        activation: type[nn.Module] = nn.SiLU,
    ):
        super().__init__()

        self.has_skip = (
            skip and all([s == 1 for s in stride]) and in_channels == out_channels
        )
        modules: list[nn.Module] = [
            ConvBnAct2d(
                in_channels, in_channels * depth_multiplier, activation=activation
            ),
            ConvBnAct2d(
                in_channels * depth_multiplier,
                in_channels * depth_multiplier,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=tuple([k // 2 for k in kernel_size]),
                groups=in_channels * depth_multiplier,
                activation=activation,
            ),
        ]
        if se_after_dw_conv:
            modules.append(
                SqueezeExcite(
                    in_channels * depth_multiplier,
                    se_ratio=se_ratio,
                    activation=activation,
                )
            )
        modules.append(
            ConvBnAct2d(
                in_channels * depth_multiplier,
                out_channels,
                activation=activation,
            ),
        )
        if not se_after_dw_conv:
            modules.append(
                SqueezeExcite(out_channels, se_ratio=se_ratio, activation=activation)
            )

        self.inv_res = nn.Sequential(*modules)
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.inv_res(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class EfficientWavegram(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stem_kernel_size: int = 3,
        stem_stride: int = 1,
        layers: list[int] = [1, 2, 2, 3],
        kernel_sizes: list[int] = [3, 3, 3, 5],
        hidden_dims: list[int] = [32, 16, 24, 40, 64],
        strides: list[int] = [2, 2, 2, 2],
        num_filter_banks: int = 32,
        depth_multiplier: int = 4,
        se_ratio: int = 16,
        se_after_dw_conv: bool = True,
        activation: type[nn.Module] = nn.SiLU,
    ):
        super().__init__()
        assert (
            hidden_dims[-1] % num_filter_banks == 0
        ), f"hidden_dims[-1] must be divisible by num_filter_banks, but got {hidden_dims[-1]} and {num_filter_banks} respectively."
        self.num_filter_banks = num_filter_banks
        self.stem_stride = stem_stride
        self.strides = strides

        self.stem_conv = ConvBnAct2d(
            in_channels,
            hidden_dims[0],
            kernel_size=(1, stem_kernel_size),
            stride=(1, stem_stride),
            activation=activation,
        )
        self.wave_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        InvertedResidual(
                            c_in if j == 0 else c_out,
                            c_out,
                            kernel_size=(1, k),
                            stride=(1, s) if j == 0 else (1, 1),
                            depth_multiplier=depth_multiplier,
                            se_ratio=se_ratio,
                            se_after_dw_conv=se_after_dw_conv,
                            activation=activation,
                        )
                        for j in range(nl)
                    ]
                )
                for i, (c_in, c_out, k, s, nl) in enumerate(
                    zip(hidden_dims, hidden_dims[1:], kernel_sizes, strides, layers)
                )
            ]
        )
        self.mapper = nn.Sequential(
            nn.Conv2d(
                hidden_dims[-1] // self.num_filter_banks,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            activation(),
        )

    @property
    def hop_length(self):
        return self.stem_stride * reduce(operator.mul, self.strides, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: b c t
        return: b c f t
        """
        x = rearrange(x, "b c t -> b c 1 t")
        x = self.stem_conv(x)
        for i, b in enumerate(self.wave_blocks):
            x = b(x)
        x = rearrange(x, "b (c f) 1 t -> b c f t", f=self.num_filter_banks)
        x = self.mapper(x)
        return x


if __name__ == "__main__":
    from torchinfo import summary

    batch_size = 2
    num_frames = 2048
    num_filter_banks = 64
    in_channels = 2
    out_channels = 1
    hidden_dims = [32, 16, 24, 40, 64]
    num_blocks = len(hidden_dims) - 1

    input = torch.randn(batch_size, in_channels, num_frames)
    model = EfficientWavegram(
        in_channels=in_channels,
        out_channels=out_channels,
        num_filter_banks=num_filter_banks,
        hidden_dims=hidden_dims,
        layers=[1, 2, 2, 3],
        kernel_sizes=[3, 3, 3, 5],
        strides=[2, 2, 2, 2],
    )
    output = model(input)
    assert output.shape == (
        in_channels,
        out_channels,
        num_filter_banks,
        num_frames / 2**num_blocks,
    ), f"Got: {output.shape}"

    summary(model, input_size=(batch_size, in_channels, num_frames))
