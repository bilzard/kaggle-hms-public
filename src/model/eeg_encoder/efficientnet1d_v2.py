"""
EfficientNet with fixed width

Implementation is similar to [1].

[1] https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/efficientnet.py
"""

import torch.nn as nn
from einops import rearrange
from timm.layers import DropPath
from torch import Tensor

from src.model.basic_block.transformer import SharedQkMultiHeadSelfAttention


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
        padding: tuple[int, ...] | str = "same",
        groups: int = 1,
        activation: type[nn.Module] = nn.SiLU,
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
        self.has_skip = (
            skip and all([s == 1 for s in stride]) and in_channels == out_channels
        )
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
        activation: type[nn.Module],
        stride: tuple[int, ...] = (1, 1),
        depth_multiplier: int = 6,
        se_ratio: int = 24,
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

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.inv_res(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dims: int, num_heads: int, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dims)
        self.mhsa = SharedQkMultiHeadSelfAttention(hidden_dims, num_heads, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.mhsa(self.norm(x))
        return x


class EfficientNet1dV2(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stem_kernel_size: int = 3,
        stem_pool_size: int = 2,
        depth_multiplier: int = 6,
        ds_conv_se_ratio: int = 4,
        ir_conv_se_ratio: int = 24,
        kernel_sizes: list[int] = [3, 3, 3, 5, 5],
        layers: list[int] | int = [1, 2, 2, 3, 3],
        hidden_dims: list[int] = [32, 16, 24, 40, 64, 64],
        strides: list[int] = [1, 2, 2, 2, 2],
        frame_offset: int = 744,  # 800 - (512 - 400) // 2
        num_frames: int = 512,
        activation=nn.SiLU,
        drop_path_rate: float = 0.0,
        use_ds_conv: bool = True,
        se_after_dw_conv: bool = False,
        use_channel_mixer: bool = False,
        channel_mixer_kernel_size: int = 3,
        mixer_type: str = "sc",
        grad_checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__()
        if isinstance(layers, int):
            layers = [layers] * len(kernel_sizes)

        self.in_channels = in_channels
        self.depth_multiplier = depth_multiplier
        self.frame_offset = frame_offset
        self.num_frames = num_frames
        self.temporal_layers = layers
        self.num_eeg_channels = in_channels // 2
        self.drop_path_rate = drop_path_rate
        self.hidden_dims = hidden_dims

        self.stem_conv = ConvBnAct2d(
            2,
            hidden_dims[0],
            kernel_size=(1, stem_kernel_size),
            stride=(1, stem_pool_size),
            padding=(0, stem_kernel_size // 2),
            activation=activation,
            drop_path_rate=drop_path_rate,
        )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        DepthWiseSeparableConv(
                            in_channels=ci if j == 0 else co,
                            out_channels=co,
                            kernel_size=(1, k),
                            stride=(1, 1),
                            activation=activation,
                            se_ratio=ds_conv_se_ratio,
                            drop_path_rate=drop_path_rate,
                            se_after_dw_conv=se_after_dw_conv,
                        )
                        if i == 0 and use_ds_conv
                        else InvertedResidual(
                            in_channels=ci if j == 0 else co,
                            out_channels=co,
                            depth_multiplier=depth_multiplier,
                            kernel_size=(1, k),
                            stride=(1, s) if j == 0 else (1, 1),
                            activation=activation,
                            se_ratio=ir_conv_se_ratio,
                            drop_path_rate=drop_path_rate,
                            se_after_dw_conv=se_after_dw_conv,
                        )
                        for j in range(nl)
                    ],
                    (
                        InvertedResidual(
                            in_channels=co,
                            out_channels=co,
                            kernel_size=(channel_mixer_kernel_size, 1),
                            activation=activation,
                            depth_multiplier=depth_multiplier,
                            se_ratio=depth_multiplier,
                            se_after_dw_conv=se_after_dw_conv,
                        )
                        if mixer_type == "ir"
                        else (
                            DepthWiseSeparableConv(
                                in_channels=co,
                                out_channels=co,
                                kernel_size=(channel_mixer_kernel_size, 1),
                                activation=activation,
                                se_ratio=depth_multiplier,
                                se_after_dw_conv=se_after_dw_conv,
                            )
                        )
                    )
                    if use_channel_mixer
                    else nn.Identity(),
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
    def out_channels(self) -> int:
        return self.hidden_dims[-1]

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (d ch b) c t
        output: (d ch b) c t
        """
        x = x[
            :, :, self.frame_offset : self.frame_offset + self.num_frames
        ]  # 中央の512 frame(12.8sec)に絞る
        x = rearrange(x, "b (c ch) t -> b c ch t", c=2)
        x = self.stem_conv(x)
        for block in self.blocks:
            x = block(x)
        x = rearrange(x, "(d b) c ch t -> (d ch b) c t", d=2, ch=self.num_eeg_channels)
        return x


if __name__ == "__main__":
    from torchinfo import summary

    batch_size = 2
    in_channels = 2
    num_eeg_channels = 10
    num_frames = 2048

    model = EfficientNet1dV2(
        in_channels=in_channels * num_eeg_channels, num_eeg_channels=num_eeg_channels
    )
    summary(model, input_size=(batch_size, in_channels * num_eeg_channels, num_frames))
