import torch
import torch.nn as nn
from torch import Tensor

from src.model.basic_block import GatedSpecAttention


class SeBlock(nn.Module):
    def __init__(self, n_channels, se_ratio):
        super().__init__()

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Conv2d(n_channels, n_channels // se_ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(n_channels // se_ratio, n_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_out = x * self.se(x)
        return x_out


class ConvBnPReLu(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        groups=1,
        conv: type[nn.Conv2d | nn.ConvTranspose2d] = nn.Conv2d,
    ):
        super().__init__()

        padding = (kernel_size - stride) // 2 if stride > 1 else "same"
        self.layers = nn.Sequential(
            conv(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,  # type: ignore
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )

    def forward(self, x):
        x_out = self.layers(x)
        return x_out


class ResBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, kernel_size, se_ratio):
        super().__init__()

        self.layers = nn.Sequential(
            ConvBnPReLu(input_size, hidden_size, kernel_size, stride=1),
            ConvBnPReLu(hidden_size, output_size, kernel_size, stride=1),
            SeBlock(output_size, se_ratio),
        )
        self.skip = (
            nn.Conv2d(input_size, hidden_size, kernel_size=1)
            if input_size != hidden_size
            else nn.Identity()
        )

    def forward(self, x):
        return self.skip(x) + self.layers(x)


class UpConv(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        skip_size: int = 0,
        stride: int = 1,
        se_ratio: int = 4,
        num_res_blocks: int = 1,
        attend_to_frequency_dim=True,
        frequency_smooth_kernel_size=3,
        temporal_smooth_kernel_size=3,
    ):
        super().__init__()

        self.up_scale = nn.Sequential(
            ConvBnPReLu(
                hidden_size, hidden_size, kernel_size, stride, conv=nn.ConvTranspose2d
            ),
        )
        conv_layers = []

        for i in range(num_res_blocks):
            input_size = hidden_size + skip_size if i == 0 else hidden_size
            conv_layers.append(
                ResBlock(
                    input_size,
                    hidden_size,
                    hidden_size,
                    kernel_size,
                    se_ratio,
                )
            )

        self.conv = nn.Sequential(*conv_layers)
        self.attention = GatedSpecAttention(
            hidden_size,
            attend_to_frequency_dim=attend_to_frequency_dim,
            frequency_smooth_kernel_size=frequency_smooth_kernel_size,
            temporal_smooth_kernel_size=temporal_smooth_kernel_size,
        )

    def forward(self, x, skip=None):
        x = self.up_scale(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = x * self.attention(x)
        return x


class UnetV3(nn.Module):
    """
    Unet with Gated Spec Attention
    """

    def __init__(
        self,
        encoder_channels: list[int],  # d0-d5
        encoder_depth: int = 4,
        hidden_size: int = 64,
        kernel_sizes: list[int] = [4, 4, 4],
        strides: list[int] = [2, 2, 2],
        se_ratio: int = 4,
        num_res_blocks: int = 1,
        attend_to_frequency_dim=True,
        frequency_smooth_kernel_size=3,
        temporal_smooth_kernel_size=3,
    ):
        super().__init__()
        self.encoder_depth = encoder_depth
        encoder_channels = encoder_channels[-encoder_depth:][::-1]  # d5, d4, d3, d2

        self.channel_mapper = nn.Conv2d(encoder_channels[0], hidden_size, kernel_size=1)
        self.up_layers = nn.ModuleList(
            UpConv(
                hidden_size,
                kernel_size=k,
                stride=s,
                skip_size=skip_size,
                se_ratio=se_ratio,
                num_res_blocks=num_res_blocks,
                attend_to_frequency_dim=attend_to_frequency_dim,
                frequency_smooth_kernel_size=frequency_smooth_kernel_size,
                temporal_smooth_kernel_size=temporal_smooth_kernel_size,
            )
            for k, s, skip_size in zip(
                kernel_sizes,
                strides,
                encoder_channels[1:],  # C4, C3, C2
            )
        )
        self._output_size = hidden_size

    @property
    def output_size(self):
        return self._output_size

    def forward(self, features: list[Tensor]) -> Tensor:
        features = features[-self.encoder_depth :][::-1]  # C5, C4, C3, C2

        x = features[0]  # C5
        x = self.channel_mapper(x)
        for up_layer, skip in zip(self.up_layers, features[1:]):  # C4, C3, C2
            x = up_layer(x, skip)

        return x
