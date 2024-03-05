"""
1D version of EfficientNet

Implementation is similar to [1].

[1] https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/efficientnet.py
"""

import torch.nn as nn
from timm.layers import DropPath
from torch import Tensor


class SqueezeExcite(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        se_ratio: int,
        activation=nn.ELU,
    ):
        super().__init__()

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Conv1d(hidden_dim, hidden_dim // se_ratio, kernel_size=1),
            activation(inplace=True),
            nn.Conv1d(hidden_dim // se_ratio, hidden_dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.se(x)


class ConvBnAct1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int | str = "same",
        groups: int = 1,
        activation=nn.ELU,
        drop_path_rate: float = 0.0,
        skip: bool = False,
    ):
        super().__init__()
        assert (
            kernel_size >= stride
        ), f"kernel_size must be greater than stride. Got {kernel_size} and {stride}."

        match padding:
            case "valid":
                padding = 0
            case "same":
                padding = (kernel_size - stride) // 2
            case _:
                pass

        self.has_skip = skip and stride == 1 and in_channels == out_channels
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,  # type: ignore
                stride,  # type: ignore
                padding,  # type: ignore
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
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
        hidden_dim: int,
        kernel_size: int,
        activation,
        se_ratio: int = 4,
        no_skip: bool = False,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.has_skip = not no_skip

        self.conv = nn.Sequential(
            ConvBnAct1d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                groups=hidden_dim,
                activation=activation,
            ),
            ConvBnAct1d(hidden_dim, hidden_dim, activation=activation),
            SqueezeExcite(hidden_dim, se_ratio=se_ratio, activation=activation),
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


class InvertedResidual(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        depth_multiplier: int,
        kernel_size: int,
        activation,
        se_ratio: int = 4,
        skip: bool = True,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.has_skip = not skip

        self.inv_res = nn.Sequential(
            ConvBnAct1d(hidden_dim, hidden_dim, activation=activation),
            ConvBnAct1d(
                hidden_dim,
                hidden_dim * depth_multiplier,
                kernel_size=kernel_size,
                groups=hidden_dim,
                activation=activation,
            ),
            ConvBnAct1d(
                hidden_dim * depth_multiplier, hidden_dim, activation=activation
            ),
            SqueezeExcite(hidden_dim, se_ratio=se_ratio, activation=activation),
        )
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.inv_res(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class ResBlock1d(nn.Module):
    def __init__(
        self,
        layer: nn.Module,
        pool_size: int = 2,
        skip: bool = True,
    ):
        super().__init__()
        self.has_skip = skip
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        self.layer = nn.Sequential(
            layer,
            nn.MaxPool1d(kernel_size=pool_size, stride=pool_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        if not self.has_skip:
            return self.layer(x)

        return self.pool(x) + self.layer(x)


class EfficientNet1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        depth_multiplier: int = 4,
        stem_kernel_size: int = 3,
        kernel_sizes: list[int] = [3, 3, 3, 5, 5],
        pool_sizes: list[int] = [2, 2, 2, 2, 2],
        layers: list[int] | int = [1, 2, 2, 3, 3],
        frame_offset: int = 744,  # 800 - (512 - 400) // 2
        num_frames: int = 512,
        skip: bool = True,
        activation=nn.ELU,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        if isinstance(layers, int):
            layers = [layers] * len(kernel_sizes)

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.depth_multiplier = depth_multiplier
        self.frame_offset = frame_offset
        self.num_frames = num_frames
        self.temporal_pool_sizes = pool_sizes
        self.temporal_layers = layers
        self.drop_path_rate = drop_path_rate

        self.stem_conv = ConvBnAct1d(
            in_channels,
            hidden_dim,
            kernel_size=stem_kernel_size,
            activation=activation,
            drop_path_rate=drop_path_rate,
        )
        self.efficient_net = nn.Sequential(
            *[
                ResBlock1d(
                    layer=nn.Sequential(
                        *[
                            DepthWiseSeparableConv(
                                hidden_dim=hidden_dim,
                                kernel_size=k,
                                activation=activation,
                                se_ratio=depth_multiplier,
                                drop_path_rate=drop_path_rate,
                            )
                            if i == 0
                            else InvertedResidual(
                                hidden_dim=hidden_dim,
                                depth_multiplier=depth_multiplier,
                                kernel_size=k,
                                activation=activation,
                                se_ratio=depth_multiplier,
                                drop_path_rate=drop_path_rate,
                            )
                            for _ in range(nl)
                        ],
                    ),
                    pool_size=p,
                    skip=skip,
                )
                for i, (k, p, nl) in enumerate(zip(kernel_sizes, pool_sizes, layers))
            ]
        )

    @property
    def out_channels(self) -> int:
        return self.hidden_dim

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (d ch b) c t
        output: (d ch b) c t
        """
        x = x[
            :, :, self.frame_offset : self.frame_offset + self.num_frames
        ]  # 中央の512 frame(12.8sec)に絞る
        x = self.stem_conv(x)
        x = self.efficient_net(x)  # b c ch t

        return x


if __name__ == "__main__":
    from torchinfo import summary

    batch_size = 40
    in_channels = 2
    num_frames = 2048

    model = EfficientNet1d(in_channels)
    summary(model, input_size=(batch_size, in_channels, num_frames))
