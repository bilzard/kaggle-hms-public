import torch.nn as nn
from einops import rearrange
from torch import Tensor

from src.model.basic_block import GruBlock, calc_similarity, vector_pair_mapping


class EegDualPerChannelFeatureProcessor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        eeg_channels: int,
        lr_mapping_type: str = "identity",
        num_gru_blocks: int = 1,
        num_sim_gru_blocks: int = 1,
        use_ff: bool = False,
        use_ff_sim: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.eeg_channels = eeg_channels
        self.lr_mapping_type = lr_mapping_type

        self.similarity_encoder = nn.Sequential(
            nn.Conv1d(1, self.hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.hidden_dim),
            nn.PReLU(),
        )
        self.gru = GruBlock(hidden_dim, n_layers=num_gru_blocks, use_ff=use_ff)
        self.gru_sim = GruBlock(
            hidden_dim, n_layers=num_sim_gru_blocks, use_ff=use_ff_sim
        )

    @property
    def out_channels(self) -> int:
        return 2 * self.in_channels + self.hidden_dim

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (d ch b) c t
        output: b c
        """
        x0 = x
        feats = []

        x = x.mean(dim=2)  # temporal pooling -> (d ch b) c
        x = rearrange(x, "(d ch b) c -> (d b) ch c", d=2, ch=self.eeg_channels)

        # 方脳の全チャンネルの特徴をmix
        # NOTE: 厳密にはsequenceじゃないが、transformerの方が良い？
        x = self.gru(x)  # (d b) ch c
        x = x.mean(dim=1)  # (d b) c
        x = rearrange(x, "(d b) c -> d b c", d=2)
        x_left, x_right = x[0], x[1]  # b c
        feats.extend(list(vector_pair_mapping(x_left, x_right, self.lr_mapping_type)))

        # Left-right similarity
        x = rearrange(x0, "(d ch b) c t -> d (ch b) c t", d=2, ch=self.eeg_channels)
        x_left, x_right = x[0], x[1]  # (ch b) c t
        lr_sim = calc_similarity(x_left, x_right)  # calc similarity per time steps
        lr_sim = self.similarity_encoder(lr_sim)
        lr_sim = rearrange(lr_sim, "(ch b) c t -> b ch c t", ch=self.eeg_channels)
        lr_sim = lr_sim.mean(dim=3)  # temporal pooling -> b ch c
        lr_sim = self.gru_sim(lr_sim)  # EEG-channel mixing -> b ch c
        lr_sim = lr_sim.mean(dim=1)  # pooling ->  b c
        feats.append(lr_sim)

        x = torch.cat(feats, dim=1)  # b c

        return x


if __name__ == "__main__":
    import torch
    from torchinfo import summary

    duality = 2
    n_eeg_channels = 10
    batch_size = 2
    hidden_dim = 64
    n_frames = 512
    model = EegDualPerChannelFeatureProcessor(
        in_channels=hidden_dim,
        hidden_dim=hidden_dim,
        eeg_channels=n_eeg_channels,
    )
    summary(
        model,
        input_size=(duality * n_eeg_channels * batch_size, hidden_dim, n_frames),
    )
