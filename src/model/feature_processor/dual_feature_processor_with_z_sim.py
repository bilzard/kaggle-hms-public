import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from src.model.basic_block import calc_similarity, vector_pair_mapping

from .base import BaseFeatureProcessor


class DualFeatureProcessorWithZSim(BaseFeatureProcessor):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        use_lr_feature: bool = True,
        use_similarity_feature: bool = True,
        use_z_similarity_feature: bool = True,
        lr_mapping_type: str = "identity",
        z_mapping_type: str = "identity",
    ):
        super().__init__(in_channels=in_channels)
        self.hidden_dim = hidden_dim
        self.use_lr_feature = use_lr_feature
        self.use_similarity_feature = use_similarity_feature
        self.use_z_similarity_feature = use_z_similarity_feature
        self.lr_mapping_type = lr_mapping_type
        self.z_mapping_type = z_mapping_type

        if use_similarity_feature:
            self.similarity_encoder = nn.Sequential(
                nn.Conv2d(1, self.hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.hidden_dim),
                nn.PReLU(),
            )
        if use_z_similarity_feature:
            self.z_similarity_encoder = nn.Sequential(
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
        if self.use_z_similarity_feature:
            out_channels += 2 * self.hidden_dim
        return out_channels

    def forward(self, inputs: dict[str, Tensor]) -> Tensor:
        """
        x: (2B, C, 5F, 2T)
        """
        x = inputs["spec"]
        feats = []

        x0 = rearrange(x, "b c (m f) t -> b c m f t", m=5)
        x = x0[:, :, :-1, :, :]  # remove z
        x = rearrange(x, "b c m f t -> b c (m f) t", m=4)

        if self.use_z_similarity_feature:
            #
            # add z-similarity
            #
            z = x0[:, :, -1, :, :].unsqueeze(2)
            z = z.expand(-1, -1, 4, -1, -1).contiguous()
            z = rearrange(z, "b c m f t -> b c (m f) t", m=4)

            z_sim = calc_similarity(z, x)
            z_sim = self.z_similarity_encoder(z_sim)
            z_sim = rearrange(z_sim, "(d b) c f t -> d b c f t", d=2)
            z_sim_left, z_sim_right = z_sim[0], z_sim[1]
            feats.extend(
                list(vector_pair_mapping(z_sim_left, z_sim_right, self.z_mapping_type))
            )

        #
        # add L/R similarity
        #
        x = rearrange(x, "(d b) c f t -> d b c f t", d=2)
        x_left, x_right = x[0], x[1]

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
