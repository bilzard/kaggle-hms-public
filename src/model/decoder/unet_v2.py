import torch
import torch.nn as nn
from torch import Tensor


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
    ):
        super().__init__()
        self.up_scale = nn.Sequential(
            ConvBnPReLu(
                hidden_size, hidden_size, kernel_size, stride, conv=nn.ConvTranspose2d
            ),
        )
        self.conv = ConvBnPReLu(
            hidden_size + skip_size, hidden_size, kernel_size, stride=1
        )

    def forward(self, x, skip=None):
        x = self.up_scale(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UnetV2(nn.Module):
    def __init__(
        self,
        encoder_channels: list[int],  # d0-d5
        encoder_depth: int = 4,
        hidden_size: int = 64,
        kernel_sizes: list[int] = [4, 4, 4],
        strides: list[int] = [2, 2, 2],
        se_ratio: int = 4,
    ):
        super().__init__()
        self.encoder_depth = encoder_depth
        encoder_channels = encoder_channels[-encoder_depth:][::-1]  # d5, d4, d3, d2

        self.channel_mapper = nn.ModuleList(
            nn.Sequential(
                SeBlock(d_in, se_ratio),
                nn.Conv2d(d_in, hidden_size, kernel_size=1),
            )
            for d_in in encoder_channels
        )

        self.up_layers = nn.ModuleList(
            UpConv(
                hidden_size,
                kernel_size=k,
                stride=s,
                skip_size=hidden_size,
            )
            for k, s in zip(
                kernel_sizes,
                strides,
            )
        )

    def forward(self, features: list[Tensor]) -> Tensor:
        features = features[-self.encoder_depth :][::-1]  # C5, C4, C3, C2
        for i, (mapper, feature) in enumerate(zip(self.channel_mapper, features)):
            features[i] = mapper(feature)

        x = features[0]  # C5
        B, C, H, W = x.shape

        for up_layer, skip in zip(self.up_layers, features[1:]):  # C4, C3, C2
            x = up_layer(x, skip)

        return x
