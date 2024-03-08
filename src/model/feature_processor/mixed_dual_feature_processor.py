import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from src.model.basic_block import (
    CosineSimilarityEncoder2d,
    GeMPool2d,
    vector_pair_mapping,
)


class MixedDualFeatureProcessor(nn.Module):
    """
    L/R-invalid mapping + Similarity +  w GeMPool
    """

    def __init__(
        self,
        in_channels_spec: int,
        in_channels_eeg: int,
        hidden_dim: int,
        num_eeg_channels: int,
        activation_spec: nn.Module,
        activation_eeg: nn.Module,
        lr_mapping_type: str = "identity",
    ):
        super().__init__()
        self.in_channels_spec = in_channels_spec
        self.in_channels_eeg = in_channels_eeg
        self.hidden_dim = hidden_dim
        self.num_eeg_channels = num_eeg_channels
        self.lr_mapping_type = lr_mapping_type
        self.spec_similarity_encoder = CosineSimilarityEncoder2d(
            hidden_dim=hidden_dim, activation=activation_spec
        )
        self.eeg_similarity_encoder = CosineSimilarityEncoder2d(
            hidden_dim=hidden_dim, activation=activation_eeg
        )
        self.spec_pool = GeMPool2d()
        self.sim_pool = GeMPool2d()

    @property
    def out_channels(self) -> int:
        return 2 * (self.in_channels_spec + self.in_channels_eeg + self.hidden_dim)

    def forward(self, inputs: dict[str, Tensor]) -> Tensor:
        """
        inputs:
        - spec: (d b) c f t
        output: b c
        """
        spec = inputs["spec"]
        spec = rearrange(spec, "(d b) c f t -> d b c f t", d=2)
        spec_left, spec_right = spec[0], spec[1]  # b c f t
        spec_feats = []
        spec_feats.extend(
            list(vector_pair_mapping(spec_left, spec_right, self.lr_mapping_type))
        )
        spec_sim = self.spec_similarity_encoder(spec_left, spec_right)
        spec_feats.append(spec_sim)
        spec = torch.cat(spec_feats, dim=1)
        spec = self.spec_pool(spec)  # b c 1 1
        spec = rearrange(spec, "b c 1 1 -> b c")

        eeg = inputs["eeg"]
        eeg = rearrange(
            eeg, "(d ch b) c t -> d b c ch t", d=2, ch=self.num_eeg_channels
        )
        eeg_left, eeg_right = eeg[0], eeg[1]
        eeg_feats = []
        eeg_feats.extend(
            list(vector_pair_mapping(eeg_left, eeg_right, self.lr_mapping_type))
        )
        eeg_sim = self.eeg_similarity_encoder(eeg_left, eeg_right)
        eeg_feats.append(eeg_sim)
        eeg = torch.cat(eeg_feats, dim=1)
        eeg = self.sim_pool(eeg)  # b c 1 1
        eeg = rearrange(eeg, "b c 1 1 -> b c")

        output = torch.cat([spec, eeg], dim=1)

        return output


if __name__ == "__main__":
    from torchinfo import summary

    duality = 2
    batch_size = 2
    eeg_channels = 10
    in_channels_spec = 320
    in_channels_eeg = 192
    hidden_dim = 64
    F_spec, T_spec = 8, 8
    T_eeg = 16

    feature_processor = MixedDualFeatureProcessor(
        in_channels_spec,
        in_channels_eeg,
        hidden_dim,
        eeg_channels,
        nn.PReLU(),
        nn.PReLU(),
        "max-min",
    )
    spec = torch.randn(duality * batch_size, in_channels_spec, F_spec, T_spec)
    eeg = torch.randn(duality * eeg_channels * batch_size, in_channels_eeg, T_eeg)
    inputs = dict(spec=spec, eeg=eeg)
    output = feature_processor(inputs)
    print(f"{output.shape=}")
    assert output.shape == (
        2,
        2 * (in_channels_spec + in_channels_eeg + hidden_dim),
    ), f"{output.shape=}"

    summary(feature_processor, input=inputs)
