import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from src.model.basic_block import (
    GeMPool2d,
    calc_similarity,
    vector_pair_mapping,
)


class EegDualPerChannelSimpleFeatureProcessor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        eeg_channels: int,
        lr_mapping_type: str = "identity",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.eeg_channels = eeg_channels
        self.lr_mapping_type = lr_mapping_type
        self.similarity_encoder = nn.Sequential(
            nn.Conv2d(1, self.hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.hidden_dim),
            nn.PReLU(),
        )
        self.pool = GeMPool2d()

    @property
    def out_channels(self) -> int:
        return 2 * self.in_channels + self.hidden_dim

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (d ch b) c t
        output: b c ch t
        """
        feats = []

        x = rearrange(x, "(d ch b) c t -> d b c ch t", d=2, ch=self.eeg_channels)
        x_left, x_right = x[0], x[1]  # b c ch t
        feats.extend(
            list(vector_pair_mapping(x_left, x_right, self.lr_mapping_type))
        )  # b 2 * c ch t

        # Left-right similarity
        lr_sim = calc_similarity(x_left, x_right)  # b 1 ch t
        lr_sim = self.similarity_encoder(lr_sim)  # b c ch t
        feats.append(lr_sim)

        x = torch.cat(feats, dim=1)  # b c ch t
        x = self.pool(x)  # b c 1 1
        x = rearrange(x, "b c 1 1 -> b c")

        return x


if __name__ == "__main__":
    import torch
    from torchinfo import summary

    duality = 2
    n_eeg_channels = 10
    batch_size = 2
    hidden_dim = 64
    n_frames = 512
    model = EegDualPerChannelSimpleFeatureProcessor(
        in_channels=hidden_dim,
        hidden_dim=hidden_dim,
        eeg_channels=n_eeg_channels,
    )
    summary(
        model,
        input_size=(duality * n_eeg_channels * batch_size, hidden_dim, n_frames),
    )
