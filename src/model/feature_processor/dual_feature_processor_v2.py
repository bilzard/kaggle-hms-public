import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from src.model.basic_block import (
    CosineSimilarityEncoder2d,
    GeMPool2d,
    vector_pair_mapping,
)
from src.model.feature_processor.base import BaseFeatureProcessor


class DualFeatureProcessorV2(BaseFeatureProcessor):
    """
    Two-stage pooling:
    1. pool frequency-temporal embedding
    2. pool EEG channel embedding
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        activation: nn.Module,
        lr_mapping_type: str = "identity",
        tile_size_per_channel: tuple[int, int] = (4, 8),
    ):
        super().__init__(in_channels=in_channels)
        self.hidden_dim = hidden_dim
        self.lr_mapping_type = lr_mapping_type
        self.tile_size_per_channel = tile_size_per_channel
        self.similarity_encoder = CosineSimilarityEncoder2d(
            hidden_dim=hidden_dim, activation=activation
        )
        self.freq_temporal_pool = GeMPool2d()
        self.channel_pool = GeMPool2d()

    @property
    def out_channels(self) -> int:
        return 2 * self.in_channels + self.hidden_dim

    def forward(self, inputs: dict[str, Tensor]) -> Tensor:
        """
        inputs:
        - spec: (d b) c f t
        output: b c
        """
        x = inputs["spec"]
        x = rearrange(
            x,
            "(d b) c (m f) (n t) -> d (m n b) c f t",
            d=2,
            m=self.tile_size_per_channel[0],
            n=self.tile_size_per_channel[1],
        )
        x_left, x_right = x[0], x[1]
        feats = []

        feats.extend(list(vector_pair_mapping(x_left, x_right, self.lr_mapping_type)))

        sim = self.similarity_encoder(x_left, x_right)
        feats.append(sim)

        x = torch.cat(feats, dim=1)

        x = self.freq_temporal_pool(x)  # (m n b) c 1 1
        x = rearrange(
            x,
            "(m n b) c 1 1 -> b c m n",
            m=self.tile_size_per_channel[0],
            n=self.tile_size_per_channel[1],
        )
        x = self.channel_pool(x)  # b c 1 1
        x = rearrange(x, "b c 1 1 -> b c")
        return x


if __name__ == "__main__":
    from torchinfo import summary

    duality = 2
    batch_size = 2
    in_channels = 128
    hidden_dim = 64
    F, T = 16, 16
    feature_processor = DualFeatureProcessorV2(in_channels, hidden_dim, nn.PReLU())
    spec = torch.randn(duality * batch_size, in_channels, F, T)
    inputs = dict(spec=spec)
    output = feature_processor(inputs)
    print(f"{output.shape=}")
    assert output.shape == (2, 2 * in_channels + hidden_dim), f"{output.shape=}"

    summary(feature_processor, input=inputs)
