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


class WaveBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 2,
        pool_size: int = 2,
        bottleneck_ratio: int = 4,
    ):
        super().__init__(
            nn.Conv1d(
                in_channels,
                out_channels // bottleneck_ratio,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels // bottleneck_ratio),
            nn.ReLU(),
            nn.Conv1d(
                out_channels // bottleneck_ratio,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=2,
                padding=((kernel_size - 1) // 2) * dilation,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size),
        )


class Wavegram(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stem_kernel_size: int = 3,
        stem_stride: int = 1,
        hidden_dims: list[int] = [64, 64, 64, 128, 128],
        num_filter_banks: int = 32,
    ):
        super().__init__()
        assert (
            hidden_dims[-1] % num_filter_banks == 0
        ), f"hidden_dims[-1] must be divisible by num_filter_banks, but got {hidden_dims[-1]} and {num_filter_banks} respectively."
        self.num_filter_banks = num_filter_banks

        self.stem_conv = nn.Sequential(
            nn.Conv1d(
                in_channels,
                hidden_dims[0],
                kernel_size=stem_kernel_size,
                stride=stem_stride,
                padding=(stem_kernel_size - stem_stride) // 2,
                bias=False,
            ),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
        )
        self.wave_blocks = nn.ModuleList(
            [
                WaveBlock(c_in, c_out)
                for c_in, c_out in zip(hidden_dims, hidden_dims[1:])
            ]
        )
        self.mapper = Conv2dBlock(
            in_channels=hidden_dims[-1] // num_filter_banks,
            out_channels=out_channels,
        )

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

    # Test Wavegram
    batch_size = 2
    num_frames = 2048
    num_filter_banks = 64
    in_channels = 2
    out_channels = 32
    hidden_dims = [64, 64, 64, 128, 128]
    num_blocks = len(hidden_dims) - 1

    input = torch.randn(batch_size, in_channels, num_frames)
    model = Wavegram(
        in_channels=in_channels,
        out_channels=out_channels,
        num_filter_banks=num_filter_banks,
        hidden_dims=hidden_dims,
    )
    output = model(input)
    assert output.shape == (
        in_channels,
        out_channels,
        num_filter_banks,
        num_frames / 2**num_blocks,
    ), f"Got: {output.shape}"

    summary(model, input_size=(batch_size, in_channels, num_frames))
