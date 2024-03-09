import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from src.model.basic_block import (
    CosineSimilarityEncoder3d,
    GeMPool3d,
    vector_pair_mapping,
)
from src.model.feature_processor.base import BaseFeatureProcessor


class DualChannelSeparatedFeatureProcessor(BaseFeatureProcessor):
    """
    L/R-invalid mapping + Similarity +  w GeMPool
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_eeg_channels: int,
        activation: nn.Module,
        lr_mapping_type: str = "identity",
    ):
        super().__init__(in_channels=in_channels)
        self.hidden_dim = hidden_dim
        self.lr_mapping_type = lr_mapping_type
        self.num_eeg_channels = num_eeg_channels
        self.similarity_encoder = CosineSimilarityEncoder3d(
            hidden_dim=hidden_dim, activation=activation
        )
        self.pool = GeMPool3d()

    @property
    def out_channels(self) -> int:
        return 2 * self.in_channels + self.hidden_dim

    def forward(self, inputs: dict[str, Tensor]) -> Tensor:
        """
        inputs:
        - spec: (d ch b) c f t
        output: b c
        """
        x = inputs["spec"]
        x = rearrange(
            x, "(d ch b) c f t -> d b c ch f t", d=2, ch=self.num_eeg_channels
        )
        x_left, x_right = x[0], x[1]  # b c ch f t
        feats = []

        feats.extend(list(vector_pair_mapping(x_left, x_right, self.lr_mapping_type)))

        sim = self.similarity_encoder(x_left, x_right)
        feats.append(sim)

        x = torch.cat(feats, dim=1)

        x = self.pool(x)  # b c 1 1 1
        x = rearrange(x, "b c 1 1 1 -> b c")
        return x


if __name__ == "__main__":
    from torchinfo import summary

    duality = 2
    batch_size = 2
    in_channels = 128
    hidden_dim = 64
    F, T = 16, 16
    num_eeg_channels = 8
    feature_processor = DualChannelSeparatedFeatureProcessor(
        in_channels, hidden_dim, num_eeg_channels, nn.PReLU()
    )
    spec = torch.randn(duality * num_eeg_channels * batch_size, in_channels, F, T)
    inputs = dict(spec=spec)
    output = feature_processor(inputs)
    print(f"{output.shape=}")
    assert output.shape == (2, 2 * in_channels + hidden_dim), f"{output.shape=}"

    summary(feature_processor, input=inputs)
