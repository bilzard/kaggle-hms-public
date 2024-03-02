import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from src.model.basic_block import calc_similarity, vector_pair_mapping

from .base import BaseFeatureProcessor


class DualFeatureProcessorWithMask(BaseFeatureProcessor):
    """
    channelのqualityを加味したfeature engineering
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        lr_mapping_type: str = "identity",
        output_stride: int = 32,
    ):
        super().__init__(in_channels=in_channels)
        self.hidden_dim = hidden_dim
        self.lr_mapping_type = lr_mapping_type
        self.output_stride = output_stride
        self.feature_dim = in_channels

        self.mask_encoder = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.PReLU(),
        )
        self.similarity_encoder = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.PReLU(),
        )
        self.downsample = nn.Upsample(
            scale_factor=1 / output_stride, align_corners=False, mode="bilinear"
        )

    @property
    def out_channels(self) -> int:
        out_channels = 2 * self.feature_dim + 2 * self.hidden_dim
        return out_channels

    def forward(self, inputs: dict[str, Tensor]) -> Tensor:
        x, spec_mask = inputs["spec"], inputs["spec_mask"]

        # map scalar mask to feature space
        spec_mask = self.downsample(spec_mask)
        mask_emb = self.mask_encoder(spec_mask)
        mask_emb = rearrange(mask_emb, "(d b) c f t -> d b c f t", d=2)
        mask_emb_left, mask_emb_right = mask_emb[0], mask_emb[1]

        # order-invariant spec feature
        # 変換適用後は左右の区別がなくなるので先にmask embeddingを加える
        x = rearrange(x, "(d b) c f t -> d b c f t", d=2)
        x_left, x_right = x[0], x[1]
        feats = list(
            vector_pair_mapping(
                x_left,
                x_right,
                self.lr_mapping_type,
            )
        )

        # spec similarity feature
        spec_sim = calc_similarity(x_left, x_right)
        spec_sim = self.similarity_encoder(spec_sim)
        feats.append(spec_sim)

        # mask similarity feature
        mask_sim = calc_similarity(mask_emb_left, mask_emb_right)
        mask_sim = self.similarity_encoder(mask_sim)
        feats.append(mask_sim)

        x = torch.cat(feats, dim=1)
        return x
