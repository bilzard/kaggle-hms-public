import torch
import torch.nn as nn
from torch import Tensor


class ConvBnRelu(nn.Module):
    def __init__(self, kernel_size: int, in_channels: int, hidden_dim: int):
        super().__init__()
        self.cbn = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.PReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.cbn(x)


class MixSkip(nn.Module):
    def __init__(
        self,
        kernel_size: int = 3,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.cbn = ConvBnRelu(
            kernel_size=kernel_size, in_channels=hidden_dim, hidden_dim=hidden_dim
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        """
        解像度の小さいfeature mapをskipに入力して混ぜる
        """
        return self.cbn(self.pool(x) + skip)


class FeatureMixer(nn.Module):
    def __init__(
        self,
        depth: int,
        name: str,
        kernel_size: int = 3,
        hidden_dim: int = 64,
        debug: bool = False,
    ):
        super().__init__()
        self.mixers = nn.ModuleList(
            [
                MixSkip(
                    kernel_size=kernel_size,
                    hidden_dim=hidden_dim,
                )
                for _ in range(depth - 1)
            ]
        )
        self.name = name
        self.debug = debug

    def forward(self, features: list[Tensor]) -> list[Tensor]:
        """
        features: [d2, d3, d4, d5]

        returns
        -------
        features: [d2, d3, d4, d5]
        """
        for i, (mixer, feat, skip) in enumerate(
            zip(self.mixers, features[:-1], features[1:])
        ):
            if self.debug:
                print(
                    f"mixer{self.name} @ features[{i}] feat: {tuple(feat.shape)} skip: {tuple(skip.shape)}"
                )
            features[i] = mixer(feat, skip)
        return features


class PoolFpn(nn.Module):
    """
    Poolingによるclassification専用のFPN
    通常のFPNがupsampleして一番大きなfeature mapにfitするのに対し、
    この実装ではpoolingにより一番小さなfeature mapにfitする
    """

    def __init__(
        self,
        encoder_channels: list[int],  # d0-d5
        kernel_size: int = 3,
        hidden_dim: int = 64,
        depth: int = 4,
        debug: bool = False,
    ):
        super().__init__()
        self.channels = encoder_channels[-depth:]
        self.mappers = nn.ModuleList(
            [
                ConvBnRelu(
                    kernel_size=1, in_channels=in_channels, hidden_dim=hidden_dim
                )
                for i, in_channels in enumerate(self.channels)
            ]
        )
        self.mixers = nn.ModuleList(
            [
                FeatureMixer(
                    depth=depth - i,
                    name=f"mixer{i}",
                    kernel_size=kernel_size,
                    hidden_dim=hidden_dim,
                    debug=debug,
                )
                for i in range(depth - 1)
            ]
        )
        self.depth = depth
        self.hidden_dim = hidden_dim
        self._output_size = self.hidden_dim * self.depth

    @property
    def output_size(self) -> int:
        return self._output_size

    def forward(self, features: list[Tensor]) -> Tensor:
        features = features[-self.depth :]
        for i, (mapper, feature) in enumerate(zip(self.mappers, features)):
            features[i] = mapper(feature)

        for mixer in self.mixers:
            features = mixer(features)

        x = torch.cat(features, dim=1)
        return x


if __name__ == "__main__":
    import torch
    import torch.nn as nn

    encoder_channels = [10, 20, 40, 80, 160, 320]
    features = []
    for i in range(len(encoder_channels), 0, -1):
        x = torch.randn(1, encoder_channels[-i], 10 * 2**i, 10 * 2**i)
        features.append(x)

    for i in range(len(features)):
        print(f"features[{i}]:", tuple(features[i].shape))

    model = PoolFpn(encoder_channels, debug=True)
    output = model(features)
    print("output:", tuple(output.shape))
