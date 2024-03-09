import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from src.model.basic_block import (
    CosineSimilarityEncoder2d,
    GeMPool2d,
    vector_pair_mapping,
)


class EegDualPerChannelFeatureProcessorV2(nn.Module):
    """
    L/R-invalid mapping + Similarity +  w GeMPool
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_eeg_channels: int,
        lr_mapping_type: str = "identity",
        activation: type[nn.Module] = nn.PReLU,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_eeg_channels = num_eeg_channels
        self.lr_mapping_type = lr_mapping_type
        self.similarity_encoder = CosineSimilarityEncoder2d(
            hidden_dim=hidden_dim, activation=activation()
        )
        self.pool = GeMPool2d()

    @property
    def out_channels(self) -> int:
        return 2 * self.in_channels + self.hidden_dim

    def forward(self, x: Tensor) -> Tensor:
        """
        inputs:
        - spec: (d b) c f t
        output: b c
        """
        x = rearrange(x, "(d ch b) c t -> d b c ch t", d=2, ch=self.num_eeg_channels)
        x_left, x_right = x[0], x[1]
        feats = []
        feats.extend(list(vector_pair_mapping(x_left, x_right, self.lr_mapping_type)))
        sim = self.similarity_encoder(x_left, x_right)
        feats.append(sim)
        x = torch.cat(feats, dim=1)  # b c ch t
        x = self.pool(x)  # b c 1 1
        x = rearrange(x, "b c 1 1 -> b c")

        return x


if __name__ == "__main__":
    from torchinfo import summary

    duality = 2
    batch_size = 2
    eeg_channels = 10
    in_channels = 64
    hidden_dim = 64
    T = 16

    feature_processor = EegDualPerChannelFeatureProcessorV2(
        in_channels,
        hidden_dim,
        eeg_channels,
        "max-min",
        activation=nn.PReLU,
    )
    eeg = torch.randn(duality * eeg_channels * batch_size, in_channels, T)
    output = feature_processor(eeg)
    print(f"{output.shape=}")

    summary(feature_processor, input=eeg)
