import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from src.model.basic_block import calc_similarity, vector_pair_mapping

from .base import BaseFeatureProcessor


class DualFeatureProcessor(BaseFeatureProcessor):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        use_lr_feature: bool = True,
        use_similarity_feature: bool = True,
        lr_mapping_type: str = "identity",
    ):
        super().__init__(in_channels=in_channels)
        self.hidden_dim = hidden_dim
        self.use_lr_feature = use_lr_feature
        self.use_similarity_feature = use_similarity_feature
        self.lr_mapping_type = lr_mapping_type

        if use_similarity_feature:
            self.similarity_encoder = nn.Sequential(
                nn.Conv2d(1, self.hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.hidden_dim),
                nn.PReLU(),
            )

    @property
    def out_channels(self) -> int:
        out_channels = 0
        if self.use_lr_feature:
            out_channels += 2 * self.in_channels
        if self.use_similarity_feature:
            out_channels += self.hidden_dim
        return out_channels

    def forward(self, inputs: dict[str, Tensor]) -> Tensor:
        """
        inputs:
        - spec: (d b) c f t
        output: b c f t
        """
        x = inputs["spec"]
        x = rearrange(x, "(d b) c f t -> d b c f t", d=2)
        x_left, x_right = x[0], x[1]  # (b c f t)
        feats = []

        if self.use_lr_feature:
            feats.extend(
                list(vector_pair_mapping(x_left, x_right, self.lr_mapping_type))
            )

        if self.use_similarity_feature:
            sim = calc_similarity(x_left, x_right)
            sim = self.similarity_encoder(sim)
            feats.append(sim)

        x = torch.cat(feats, dim=1)
        return x
