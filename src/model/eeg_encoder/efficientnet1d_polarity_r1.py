"""
1D version of EfficientNet

Implementation is similar to [1].

[1] https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/efficientnet.py

NOTE: Implementing with 2D conv is 25~30% faster than 1D conv.
1. 1D conv: 20.0 step/epoch (RTX 4090)
2. 2D conv: 25.3 step/epoch (RTX 4090)
"""

import torch
import torch.nn as nn
from einops import rearrange
from timm.layers import DropPath
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from src.model.basic_block import CosineSimilarityEncoder2d
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
        momentum: float = 0.1,
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
            nn.BatchNorm2d(out_channels, momentum=momentum),
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
        kernel_size: tuple[int, ...],
        activation: type[nn.Module],
        se_ratio: int = 4,
        skip: bool = True,
        drop_path_rate: float = 0.0,
        se_after_dw_conv: bool = False,
        momentum: float = 0.1,
    ):
        super().__init__()
        self.has_skip = skip
        modules: list[nn.Module] = [
            ConvBnAct2d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                groups=hidden_dim,
                activation=activation,
                momentum=momentum,
            )
        ]
        if se_after_dw_conv:
            modules.append(
                SqueezeExcite(hidden_dim, se_ratio=se_ratio, activation=activation)
            )
        modules.append(
            ConvBnAct2d(
                hidden_dim, hidden_dim, activation=activation, momentum=momentum
            )
        )
        if not se_after_dw_conv:
            modules.append(
                SqueezeExcite(hidden_dim, se_ratio=se_ratio, activation=activation)
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
        hidden_dim: int,
        depth_multiplier: int,
        kernel_size: tuple[int, ...],
        activation: type[nn.Module],
        se_ratio: int = 4,
        skip: bool = True,
        drop_path_rate: float = 0.0,
        se_after_dw_conv: bool = False,
        momentum: float = 0.1,
    ):
        super().__init__()
        self.has_skip = skip

        modules: list[nn.Module] = [
            ConvBnAct2d(
                hidden_dim, hidden_dim, activation=activation, momentum=momentum
            ),
            ConvBnAct2d(
                hidden_dim,
                hidden_dim * depth_multiplier,
                kernel_size=kernel_size,
                groups=hidden_dim,
                activation=activation,
                momentum=momentum,
            ),
        ]
        if se_after_dw_conv:
            modules.append(
                SqueezeExcite(
                    hidden_dim * depth_multiplier,
                    se_ratio=se_ratio,
                    activation=activation,
                )
            )
        modules.append(
            ConvBnAct2d(
                hidden_dim * depth_multiplier,
                hidden_dim,
                activation=activation,
                momentum=momentum,
            ),
        )
        if not se_after_dw_conv:
            modules.append(
                SqueezeExcite(hidden_dim, se_ratio=se_ratio, activation=activation)
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


class ResBlock2d(nn.Module):
    def __init__(
        self,
        layer: nn.Module,
        pool_size: int = 2,
        skip: bool = True,
    ):
        super().__init__()
        self.has_skip = skip
        self.pool = nn.MaxPool2d(kernel_size=(1, pool_size), stride=(1, pool_size))
        self.layer = nn.Sequential(
            layer,
            nn.MaxPool2d(kernel_size=(1, pool_size), stride=(1, pool_size)),
        )

    def forward(self, x: Tensor) -> Tensor:
        if not self.has_skip:
            return self.layer(x)

        return self.pool(x) + self.layer(x)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dims: int, num_heads: int, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dims)
        self.mhsa = SharedQkMultiHeadSelfAttention(hidden_dims, num_heads, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.mhsa(self.norm(x))
        return x


class TransformerChannelMixer(nn.Module):
    def __init__(self, hidden_dims: int, num_heads: int, **kwargs):
        super().__init__()
        self.transformer = TransformerBlock(hidden_dims, num_heads, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        T = x.shape[-1]
        x = rearrange(x, "b c ch t -> (b t) ch c")
        x = self.transformer(x)
        x = rearrange(x, "(b t) ch c -> b c ch t", t=T)
        return x


class EfficientNet1dPolarityR1(nn.Module):
    """
    極性の反転の特徴を入力する。

    例) Fp1-F7 <-> inv(F7-T3)の時間ごとの類似度をsliding windowに区切って計算し、stemの出力と結合する

    本来adapterでやる処理だが、効くかどうかわからないのでお試し版。
    入力チャネルは10chで固定する。

    reversed polarityを計算する際のwindowの設計には以下の2パターンがある。
    1. k=s non-duplicated window
    2. k>s duplicated window

    revisions:
    - r1: late fusion
    """

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
        skip_in_block: bool = True,
        skip_in_layer: bool = True,
        activation=nn.ELU,
        drop_path_rate: float = 0.0,
        use_ds_conv: bool = True,
        se_after_dw_conv: bool = False,
        use_channel_mixer: bool = False,
        channel_mixer_kernel_size: int = 3,
        mixer_type: str = "sc",
        transformer_merge_type: str = "add",
        input_mask: bool = True,
        momentum: float = 0.1,
        grad_checkpointing: bool = False,
        input_planes: int = 1,
        polarity_window_size: int = 32,
        polarity_stride: int = 4,
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
        self.real_in_channels = 2 * input_planes if input_mask else input_planes
        self.num_eeg_channels = in_channels // self.real_in_channels
        self.momentum = momentum
        self.grad_checkpointing = grad_checkpointing

        self.polarity_encode = ConvBnAct2d(
            self.real_in_channels,
            hidden_dim,
            kernel_size=(1, polarity_window_size),
            stride=(1, polarity_stride),
            activation=activation,
            drop_path_rate=drop_path_rate,
            momentum=momentum,
        )
        self.sim_encode_p0 = CosineSimilarityEncoder2d(hidden_dim, nn.PReLU(), 1)
        self.sim_encode_p1 = CosineSimilarityEncoder2d(hidden_dim, nn.PReLU(), 1)
        self.stem_conv = ConvBnAct2d(
            self.real_in_channels,
            hidden_dim,
            kernel_size=(1, stem_kernel_size),
            activation=activation,
            drop_path_rate=drop_path_rate,
            momentum=momentum,
        )
        self.feat_mapper = ConvBnAct2d(
            self.hidden_dim * 3,
            hidden_dim,
            kernel_size=(1, 1),
            activation=activation,
            drop_path_rate=drop_path_rate,
            momentum=momentum,
        )
        self.efficient_net = nn.Sequential(
            *[
                ResBlock2d(
                    layer=nn.Sequential(
                        *[
                            DepthWiseSeparableConv(
                                hidden_dim=hidden_dim,
                                kernel_size=(1, k),
                                activation=activation,
                                se_ratio=depth_multiplier,
                                drop_path_rate=drop_path_rate,
                                skip=skip_in_layer,
                                se_after_dw_conv=se_after_dw_conv,
                                momentum=momentum,
                            )
                            if i == 0 and use_ds_conv
                            else InvertedResidual(
                                hidden_dim=hidden_dim,
                                depth_multiplier=depth_multiplier,
                                kernel_size=(1, k),
                                activation=activation,
                                se_ratio=depth_multiplier,
                                drop_path_rate=drop_path_rate,
                                skip=skip_in_layer,
                                se_after_dw_conv=se_after_dw_conv,
                                momentum=momentum,
                            )
                            for _ in range(nl)
                        ],
                        (
                            InvertedResidual(
                                hidden_dim=hidden_dim,
                                kernel_size=(channel_mixer_kernel_size, 1),
                                activation=activation,
                                depth_multiplier=depth_multiplier,
                                se_ratio=depth_multiplier,
                                drop_path_rate=drop_path_rate,
                                se_after_dw_conv=se_after_dw_conv,
                                momentum=momentum,
                            )
                            if mixer_type == "ir"
                            else (
                                DepthWiseSeparableConv(
                                    hidden_dim=hidden_dim,
                                    kernel_size=(channel_mixer_kernel_size, 1),
                                    activation=activation,
                                    se_ratio=depth_multiplier,
                                    drop_path_rate=drop_path_rate,
                                    se_after_dw_conv=se_after_dw_conv,
                                    momentum=momentum,
                                )
                                if mixer_type == "sc"
                                else TransformerChannelMixer(
                                    hidden_dim,
                                    num_heads=depth_multiplier,
                                    merge_type=transformer_merge_type,
                                )
                            )
                        )
                        if use_channel_mixer
                        else nn.Identity(),
                    ),
                    pool_size=p,
                    skip=skip_in_block,
                )
                for i, (k, p, nl) in enumerate(zip(kernel_sizes, pool_sizes, layers))
            ]
        )

    @torch.jit.unused
    def set_grad_checkpointing(self, enable: bool = True):
        self.grad_checkpointing = enable

    @property
    def out_channels(self) -> int:
        return 3 * self.hidden_dim

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (d ch b) c t
        output: (d ch b) c t
        """
        x = x[
            :, :, self.frame_offset : self.frame_offset + self.num_frames
        ]  # 中央の512 frame(12.8sec)に絞る
        x = rearrange(x, "b (c ch) t -> b c ch t", c=self.real_in_channels)

        b, c, ch, t = x.shape
        assert ch == 11, f"Expected 10 channels, but got {ch} channels."

        p0_indices = [4, 0, 1, 2, 0, 4, 5, 6, 10, 8, 9]
        p1_indices = [1, 2, 3, 7, 5, 6, 7, 3, 9, 10, 8]
        p0_polarity = [1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1]
        p1_polarity = [-1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1]
        x_p0 = x[:, :, p0_indices, :].clone()
        x_p1 = x[:, :, p1_indices, :].clone()
        x_p0 = x_p0 * torch.tensor(p0_polarity, device=x.device).reshape(1, 1, -1, 1)
        x_p1 = x_p1 * torch.tensor(p1_polarity, device=x.device).reshape(1, 1, -1, 1)

        x_feat = torch.cat([x, x_p0, x_p1], dim=0)  # (3 b) c ch t
        x = self.stem_conv(x_feat)

        for layer in self.efficient_net:
            if self.grad_checkpointing:
                # save 20% VRAM, but 20% slower
                x = checkpoint(layer, x, preserve_rng_state=True, use_reentrant=False)  # type: ignore
            else:
                x = layer(x)  # b c ch t
        x_src = x[:b]
        x_p0 = x[b : 2 * b]
        x_p1 = x[2 * b :]
        sim_p0 = self.sim_encode_p0(x_src, x_p0)
        sim_p1 = self.sim_encode_p1(x_src, x_p1)
        x = torch.cat([x_src, sim_p0, sim_p1], dim=1)

        x = rearrange(x, "(d b) c ch t -> (d ch b) c t", d=2, ch=self.num_eeg_channels)
        return x


if __name__ == "__main__":
    from torchinfo import summary

    batch_size = 2
    in_channels = 22
    num_frames = 2048

    model = EfficientNet1dPolarityR1(in_channels)
    summary(model, input_size=(batch_size, in_channels, num_frames))
