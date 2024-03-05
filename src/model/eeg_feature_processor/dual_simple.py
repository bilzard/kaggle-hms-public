import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from src.model.basic_block import (
    GeMPool1d,
    calc_similarity,
    vector_pair_mapping,
)


class EegDualSimpleFeatureProcessor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        lr_mapping_type: str = "identity",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.lr_mapping_type = lr_mapping_type

        self.mapper = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.similarity_encoder = nn.Sequential(
            nn.Conv1d(1, self.hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = GeMPool1d()

    @property
    def out_channels(self) -> int:
        return 3 * self.hidden_dim

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (d b) c t
        output: b c
        """
        x = self.mapper(x)

        feats = []
        x = rearrange(x, "(d b) c t -> d b c t", d=2)
        x_left, x_right = x[0], x[1]  # b c t
        feats.extend(
            list(vector_pair_mapping(x_left, x_right, self.lr_mapping_type))
        )  # b 2 * c t

        # Left-right similarity
        lr_sim = calc_similarity(x_left, x_right)  # b 1 t
        lr_sim = self.similarity_encoder(lr_sim)  # b c t
        feats.append(lr_sim)

        x = torch.cat(feats, dim=1)  # b c t
        x = self.pool(x)  # b c 1
        x = rearrange(x, "b c 1 -> b c")

        return x


if __name__ == "__main__":
    import torch
    from torchinfo import summary

    duality = 2
    batch_size = 2
    hidden_dim = 64
    n_frames = 512
    model = EegDualSimpleFeatureProcessor(
        in_channels=hidden_dim,
        hidden_dim=hidden_dim,
    )
    summary(
        model,
        input_size=(duality * batch_size, hidden_dim, n_frames),
    )
