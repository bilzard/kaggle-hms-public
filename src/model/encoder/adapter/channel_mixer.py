import torch.nn as nn
from einops import rearrange
from torch import Tensor

from src.model.basic_block.conv_block import ConvBnAct2d


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


class ChannelMixer(nn.Module):
    def __init__(
        self,
        num_eeg_channels: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        bottleneck_ratio: int = 4,
        activation: type[nn.Module] = nn.ELU,
        tensor_arrangement: str = "dual_channel_separated",
    ):
        super().__init__()
        self.num_eeg_channels = num_eeg_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mixer = nn.Sequential(
            ConvBnAct2d(
                in_channels=in_channels,
                out_channels=out_channels // bottleneck_ratio,
                activation=activation,
            ),
            ConvBnAct2d(
                in_channels=out_channels // bottleneck_ratio,
                out_channels=out_channels // bottleneck_ratio,
                kernel_size=(kernel_size, 1),
                groups=out_channels // bottleneck_ratio,
                activation=activation,
            ),
            ConvBnAct2d(
                in_channels=out_channels // bottleneck_ratio,
                out_channels=out_channels,
                activation=activation,
            ),
            SqueezeExcite(
                hidden_dim=out_channels,
                se_ratio=bottleneck_ratio,
                activation=activation,
            ),
        )
        self.tensor_arrangement = tensor_arrangement

    def pre_arrangement(self, x: Tensor) -> Tensor:
        match self.tensor_arrangement:
            case "dual_channel_separated":
                x = rearrange(
                    x,
                    "(d ch b) c f t -> (d b) c ch (f t)",
                    d=2,
                    ch=self.num_eeg_channels,
                )
                return x
            case _:
                raise NotImplementedError

    def post_arrangement(self, x: Tensor, original_shape: tuple[int, ...]) -> Tensor:
        match self.tensor_arrangement:
            case "dual_channel_separated":
                _, _, f, t = original_shape
                x = rearrange(
                    x,
                    "(d b) c ch (f t) -> (d ch b) c f t",
                    d=2,
                    ch=self.num_eeg_channels,
                    f=f,
                    t=t,
                )
                return x
            case _:
                raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (d ch b) c f t
        """
        shortcut = x
        original_shape = x.shape
        x = self.pre_arrangement(x)
        x = self.mixer(x)
        x = self.post_arrangement(x, original_shape)
        return x + shortcut
