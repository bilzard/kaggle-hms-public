"""
Pytorch implementation of EEGNet[1] by @bilzard.
The original code in Keras is available [2].

[1] EEGNet: A Compact Convolutional Neural Networkfor EEG-based Brain-Computer Interfaces
[2] https://github.com/vlawhern/arl-eegmodels
"""

import torch.nn as nn
from einops import rearrange
from torch import Tensor


class SeBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        se_ratio: int,
        activation=nn.ELU,
    ):
        super().__init__()

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(hidden_dim, hidden_dim // se_ratio, kernel_size=1),
            activation(inplace=True),
            nn.Conv2d(hidden_dim // se_ratio, hidden_dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class ConvBnAct2d(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, ...] = (1, 1),
        stride: tuple[int, ...] = (1, 1),
        padding: tuple[int, ...] | str = "same",
        groups: int = 1,
        activation=nn.ELU,
    ):
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

        super().__init__(
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


class SepConv(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        depth_multiplier: int,
        kernel_size: tuple[int, ...],
        activation,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBnAct2d(hidden_dim, hidden_dim, activation=activation),
            ConvBnAct2d(
                hidden_dim,
                hidden_dim * depth_multiplier,
                kernel_size=kernel_size,
                groups=hidden_dim,
                activation=activation,
            ),
            ConvBnAct2d(
                hidden_dim * depth_multiplier, hidden_dim, activation=activation
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class ResBlock2d(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        depth_multiplier: int,
        kernel_size: tuple[int, ...],
        temporal_pool_size: int = 2,
        se_ratio: int = 4,
        residual: bool = True,
        num_layers: int = 2,
        activation=nn.ELU,
    ):
        super().__init__()
        self.residual = residual
        self.pool = nn.AvgPool2d(
            kernel_size=(1, temporal_pool_size), stride=(1, temporal_pool_size)
        )
        self.conv = nn.Sequential(
            *[
                SepConv(
                    hidden_dim,
                    depth_multiplier,
                    kernel_size,
                    activation=activation,
                )
                for _ in range(num_layers)
            ],
            SeBlock(hidden_dim, se_ratio=se_ratio, activation=activation),
            nn.AvgPool2d(
                kernel_size=(1, temporal_pool_size), stride=(1, temporal_pool_size)
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        if not self.residual:
            return self.conv(x)

        return self.pool(x) + self.conv(x)


class EegNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        depth_multiplier: int = 4,
        stem_temporal_kernel_size: int = 65,
        temporal_kernel_sizes: list[int] = [3, 3, 5, 5, 3],
        temporal_pool_sizes: list[int] = [2, 2, 2, 2, 2],
        temporal_layers: list[int] | int = [2, 2, 2, 2, 2],
        frame_offset: int = 744,  # 800 - (512 - 400) // 2
        num_frames: int = 512,
        residual: bool = True,
        se_ratio: int = 4,
        activation=nn.ELU,
        use_channel_mixer: bool = True,
    ):
        super().__init__()
        if isinstance(temporal_layers, int):
            temporal_layers = [temporal_layers] * len(temporal_kernel_sizes)

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.depth_multiplier = depth_multiplier
        self.frame_offset = frame_offset
        self.num_frames = num_frames
        self.temporal_pool_sizes = temporal_pool_sizes
        self.temporal_layers = temporal_layers
        self.num_eeg_channels = in_channels // 2
        self.residual = residual
        self.se_ratio = se_ratio
        self.use_channel_mixer = use_channel_mixer

        self.stem_conv = ConvBnAct2d(
            2,
            hidden_dim,
            kernel_size=(1, stem_temporal_kernel_size),
            activation=activation,
        )
        self.temporal_mixer = nn.Sequential(
            *[
                ResBlock2d(
                    hidden_dim,
                    depth_multiplier,
                    (1, k),
                    residual=residual,
                    temporal_pool_size=p,
                    num_layers=nl,
                    se_ratio=se_ratio,
                    activation=activation,
                )
                for k, p, nl in zip(
                    temporal_kernel_sizes, temporal_pool_sizes, temporal_layers
                )
            ]
        )
        if use_channel_mixer:
            self.eeg_channel_mixer = nn.Sequential(
                ConvBnAct2d(
                    hidden_dim,
                    hidden_dim * depth_multiplier,
                    kernel_size=(self.num_eeg_channels, 1),
                    groups=hidden_dim,
                    padding="valid",
                    activation=activation,
                ),
                ConvBnAct2d(
                    hidden_dim * depth_multiplier, hidden_dim, activation=activation
                ),
            )

    @property
    def out_channels(self) -> int:
        return self.hidden_dim

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, C, T)
        output: (B, C, T) if use_channel_mixer else (B, C, Ch, T)
        """
        x = x[
            :, :, self.frame_offset : self.frame_offset + self.num_frames
        ]  # 中央の512 frame(12.8sec)に絞る
        x = rearrange(x, "b (c ch) t -> b c ch t", c=2)
        x = self.stem_conv(x)
        x = self.temporal_mixer(x)  # b c ch t

        if self.use_channel_mixer:
            x = self.eeg_channel_mixer(x)  # b c 1 t
            x = rearrange(x, "b c 1 t -> b c t")
        else:
            # NOTE: channel方向の信号は混合してないので後段の処理はper-channel系のモジュールに任せる
            x = rearrange(
                x, "(d b) c ch t -> (d ch b) c t", d=2, ch=self.num_eeg_channels
            )

        return x


if __name__ == "__main__":
    from torchinfo import summary

    batch_size = 4
    in_channels = 20
    num_frames = 2048

    model = EegNet(in_channels)
    summary(model, input_size=(batch_size, in_channels, num_frames))
