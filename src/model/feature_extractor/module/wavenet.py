import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor


class MaxAvgPool2d(nn.Module):
    def __init__(self, pool_size: tuple[int, int]):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=pool_size)
        self.avg_pool = nn.AvgPool2d(kernel_size=pool_size)

    def forward(self, x: Tensor) -> Tensor:
        return (self.max_pool(x) + self.avg_pool(x)) / 2.0


def pool_factory(pool_type: str, pool_size: tuple[int, int]):
    match pool_type:
        case "max":
            pool = nn.MaxPool2d(kernel_size=pool_size)
        case "avg":
            pool = nn.AvgPool2d(kernel_size=pool_size)
        case "maxavg":
            pool = MaxAvgPool2d(pool_size)
        case _:
            raise ValueError(f"Invalid pool_type: {pool_type}")
    return pool


class Conv2dBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] = (3, 3),
        pool_size: tuple[int, int] = (1, 1),
        pool_type: str = "avg",
        bottleneck_ratio: int = 4,
    ):
        padding = tuple((k - 1) // 2 for k in kernel_size)
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels // bottleneck_ratio,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,  # type: ignore
                bias=False,
            ),
            nn.BatchNorm2d(out_channels // bottleneck_ratio),
            nn.ReLU(),
            nn.Conv2d(
                out_channels // bottleneck_ratio,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,  # type: ignore
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            pool_factory(pool_type, pool_size)
            if any([p > 1 for p in pool_size])
            else nn.Identity(),
        )


class WaveBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int = 3,
        pool_size: int = 2,
        bottleneck_ratio: int = 4,
        dilation: int = 1,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                hidden_dim,
                hidden_dim // bottleneck_ratio,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
                padding="same",
                bias=False,
            ),
            nn.GELU(),
            nn.Conv1d(
                hidden_dim // bottleneck_ratio,
                hidden_dim,
                kernel_size=kernel_size,
                stride=1,
                dilation=1,
                padding="same",
                bias=False,
            ),
            nn.Tanh(),
        )
        self.gate = nn.Sequential(
            nn.Conv1d(
                hidden_dim,
                hidden_dim // bottleneck_ratio,
                kernel_size=kernel_size,
                stride=1,
                dilation=2,
                padding="same",
                bias=False,
            ),
            nn.GELU(),
            nn.Conv1d(
                hidden_dim // bottleneck_ratio,
                hidden_dim,
                kernel_size=kernel_size,
                stride=1,
                dilation=1,
                padding="same",
                bias=False,
            ),
            nn.Sigmoid(),
        )
        self.pool = nn.MaxPool1d(kernel_size=pool_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x) * self.gate(x)
        x = self.pool(x)
        return x


class WaveNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stem_kernel_size: int = 3,
        stem_stride: int = 1,
        hidden_dim: int = 64,
        num_filter_banks: int = 16,
        num_blocks: int = 4,
        bottleneck_ratio: int = 8,
    ):
        super().__init__()
        self.num_filter_banks = num_filter_banks
        self.stem_stride = stem_stride
        self._out_channels = out_channels

        self.stem_conv = nn.Sequential(
            nn.Conv1d(
                in_channels,
                hidden_dim,
                kernel_size=stem_kernel_size,
                padding=(stem_kernel_size - 1) // 2,
                bias=False,
            ),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
            nn.MaxPool1d(kernel_size=stem_stride) if stem_stride > 1 else nn.Identity(),
        )
        self.wave_blocks = nn.ModuleList(
            [
                WaveBlock(
                    hidden_dim,
                    hidden_dim,
                    bottleneck_ratio=bottleneck_ratio,
                    dilation=2**i,
                )
                for i in range(num_blocks)
            ]
        )
        self.mapper = nn.Sequential(
            nn.Conv2d(
                hidden_dim // self.num_filter_banks,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def hop_length(self):
        return self.stem_stride * 2 ** (len(self.hidden_dims) - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: b c t
        return: b c f t
        """
        x = self.stem_conv(x)
        for i, b in enumerate(self.wave_blocks):
            x = b(x)
        x = rearrange(x, "b (c f) t -> b c f t", f=self.num_filter_banks)
        x = self.mapper(x)
        return x


if __name__ == "__main__":
    from torchinfo import summary

    batch_size = 2
    num_frames = 2048
    num_filter_banks = 16
    in_channels = 2
    out_channels = 8
    hidden_dim = 64
    num_blocks = 4
    bottleneck_ratio = 8

    input = torch.randn(batch_size, in_channels, num_frames)
    model = WaveNet(
        in_channels=in_channels,
        out_channels=out_channels,
        num_filter_banks=num_filter_banks,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
    )
    output = model(input)
    assert output.shape == (
        in_channels,
        out_channels,
        num_filter_banks,
        num_frames / 2**num_blocks,
    ), f"Got: {output.shape}"

    summary(model, input_size=(batch_size, in_channels, num_frames))
