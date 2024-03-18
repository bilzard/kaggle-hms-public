import torch.nn as nn


class Mlp(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bottleneck_ratio: int = 4,
        activation: type[nn.Module] = nn.PReLU,
    ):
        super().__init__(
            nn.Linear(
                in_channels,
                in_channels // bottleneck_ratio,
                bias=False,
            ),
            nn.BatchNorm1d(in_channels // bottleneck_ratio),
            activation(),
            nn.Linear(
                in_channels // bottleneck_ratio,
                out_channels,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            activation(),
        )


if __name__ == "__main__":
    import torch
    from torchinfo import summary

    # test MLP
    batch_size = 2
    in_channels = 192
    out_channels = 64
    input = torch.randn(batch_size, in_channels)
    model = Mlp(in_channels, out_channels)
    output = model(input)
    assert output.shape == (batch_size, out_channels)

    summary(model, input_size=(batch_size, in_channels))
