import torch.nn as nn
from timm.layers import DropPath
from torch import Tensor


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
        activation_params: dict = {},
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
            activation(**activation_params)
            if activation is not None
            else nn.Identity(),
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
