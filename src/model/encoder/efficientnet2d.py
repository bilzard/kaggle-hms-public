"""
EfficientNet with fixed width

Implementation is similar to [1].

[1] https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/efficientnet.py
"""

import torch.nn as nn
from timm.layers import DropPath
from torch import Tensor

from src.model.basic_block.transformer import SharedQkMultiHeadSelfAttention


class SqueezeExcite(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        se_ratio: int,
        activation: type[nn.Module] = nn.ELU,
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
        padding: tuple[int, ...] | str = "same",
        groups: int = 1,
        activation: type[nn.Module] = nn.ELU,
        drop_path_rate: float = 0.0,
        skip: bool = False,
    ):
        super().__init__()
        assert (
            kernel_size >= stride
        ), f"kernel_size must be greater than stride. Got {kernel_size} and {stride}."

        match padding:
            case "valid":
                padding = (0, 0)
            case "same":
                padding = tuple([(k - s) // 2 for k, s in zip(kernel_size, stride)])
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
        se_ratio: int = 4,
        skip: bool = True,
        drop_path_rate: float = 0.0,
        se_after_dw_conv: bool = False,
    ):
        super().__init__()
        self.has_skip = skip
        modules: list[nn.Module] = [
            ConvBnAct2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
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
                stride=stride,
                padding=tuple([k // 2 for k in kernel_size]),
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
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)),
            nn.MaxPool2d(kernel_size=stride, stride=stride),
        )

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.conv(x)
        if self.has_skip:
            x = self.drop_path(x) + self.shortcut(shortcut)
        return x


class InvertedResidual(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, ...],
        activation: type[nn.Module],
        stride: tuple[int, ...] = (1, 1),
        depth_multiplier: int = 6,
        se_ratio: int = 24,
        skip: bool = True,
        drop_path_rate: float = 0.0,
        se_after_dw_conv: bool = False,
    ):
        super().__init__()
        self.has_skip = skip

        modules: list[nn.Module] = [
            ConvBnAct2d(
                in_channels, in_channels * depth_multiplier, activation=activation
            ),
            ConvBnAct2d(
                in_channels * depth_multiplier,
                in_channels * depth_multiplier,
                kernel_size=kernel_size,
                stride=stride,
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
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)),
            nn.MaxPool2d(kernel_size=stride, stride=stride),
        )

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.inv_res(x)
        if self.has_skip:
            x = self.drop_path(x) + self.shortcut(shortcut)
        return x


class ResBlock2d(nn.Module):
    def __init__(
        self,
        layer: nn.Module,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, ...] = (1, 1),
        skip: bool = True,
    ):
        super().__init__()
        self.has_skip = skip
        self.shortcut = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1, 1),
            ),
            nn.MaxPool2d(kernel_size=stride, stride=stride),
        )
        self.layer = nn.Sequential(layer)

    def forward(self, x: Tensor) -> Tensor:
        if not self.has_skip:
            return self.layer(x)

        return self.shortcut(x) + self.layer(x)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dims: int, num_heads: int, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dims)
        self.mhsa = SharedQkMultiHeadSelfAttention(hidden_dims, num_heads, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.mhsa(self.norm(x))
        return x


class EfficientNet2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        ds_conv_se_ratio: int = 4,
        depth_multiplier: int = 6,
        ir_conv_se_ratio: int = 24,
        kernel_sizes: list[int] = [3, 3, 3, 5, 5],
        pool_sizes: list[int] = [2, 2, 2, 2, 2],
        layers: list[int] | int = [1, 2, 2, 3, 3],
        hidden_dims: list[int] = [32, 16, 24, 40, 80],
        strides: list[int] = [1, 2, 2, 2, 2],
        skip_in_block: bool = True,
        skip_in_layer: bool = True,
        activation=nn.ELU,
        drop_path_rate: float = 0.0,
        use_ds_conv: bool = True,
        se_after_dw_conv: bool = False,
        grad_checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__()
        if isinstance(layers, int):
            layers = [layers] * len(kernel_sizes)

        self.in_channels = in_channels
        self.depth_multiplier = depth_multiplier
        self.temporal_pool_sizes = pool_sizes
        self.temporal_layers = layers
        self.num_eeg_channels = in_channels // 2
        self.drop_path_rate = drop_path_rate
        self.hidden_dims = hidden_dims

        self.stem_conv = ConvBnAct2d(
            in_channels,
            hidden_dims[0],
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            activation=activation,
            drop_path_rate=drop_path_rate,
        )
        self.efficient_net = nn.Sequential(
            *[
                ResBlock2d(
                    layer=nn.Sequential(
                        *[
                            DepthWiseSeparableConv(
                                in_channels=ci if j == 0 else co,
                                out_channels=co,
                                kernel_size=(k, k),
                                activation=activation,
                                se_ratio=ds_conv_se_ratio,
                                drop_path_rate=drop_path_rate,
                                skip=skip_in_layer,
                                se_after_dw_conv=se_after_dw_conv,
                            )
                            if i == 0 and use_ds_conv
                            else InvertedResidual(
                                in_channels=ci if j == 0 else co,
                                out_channels=co,
                                depth_multiplier=depth_multiplier,
                                kernel_size=(k, k),
                                stride=(s, s) if j == 0 else (1, 1),
                                activation=activation,
                                se_ratio=ir_conv_se_ratio,
                                drop_path_rate=drop_path_rate,
                                skip=skip_in_layer,
                                se_after_dw_conv=se_after_dw_conv,
                            )
                            for j in range(nl)
                        ],
                    ),
                    in_channels=ci,
                    out_channels=co,
                    stride=(1, 1) if i == 0 else (2, 2),
                    skip=skip_in_block,
                )
                for i, (k, s, nl, ci, co) in enumerate(
                    zip(
                        kernel_sizes,
                        strides,
                        layers,
                        hidden_dims[:-1],
                        hidden_dims[1:],
                    )
                )
            ]
        )

    @property
    def out_channels(self) -> list[int]:
        return self.hidden_dims

    def forward(self, x: Tensor) -> list[Tensor]:
        """
        x: (d ch b) c f t
        output: (d ch b) c f t
        """
        x = self.stem_conv(x)
        x = self.efficient_net(x)  # b c ch t
        return [x]


if __name__ == "__main__":
    from torchinfo import summary

    batch_size = 40
    in_channels = 2
    num_filters = 64
    num_frames = 128

    model = EfficientNet2d(in_channels)
    summary(model, input_size=(batch_size, in_channels, num_filters, num_frames))
