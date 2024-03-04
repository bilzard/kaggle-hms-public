import torch.nn as nn
from torch import Tensor


class GruBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        bidirectional: bool = True,
        bottleneck_ratio: int = 4,
        use_ff: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.use_ff = use_ff

        self.norm = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        directional_factor = 2 if bidirectional else 1
        self.map = nn.Sequential(
            nn.Linear(hidden_dim * directional_factor, hidden_dim),
            nn.GELU() if use_ff else nn.Identity(),
        )
        if use_ff:
            self.ff = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim // bottleneck_ratio),
                nn.GELU(),
                nn.Linear(hidden_dim // bottleneck_ratio, hidden_dim),
            )

    def forward(self, x: Tensor, h: Tensor | None = None) -> Tensor:
        """
        Transformer-like GRU block.

        x: (B, T, C)
        output: (B, T, C)
        """
        x = x + self.map(self.gru(self.norm(x), h)[0])
        if self.use_ff:
            x = x + self.ff(x)
        return x


class GruDecoder(nn.Module):
    def __init__(self, num_blocks: int = 1, **kwargs):
        super().__init__()
        self.gru_blocks = nn.ModuleList([GruBlock(**kwargs) for _ in range(num_blocks)])

    def forward(self, x: Tensor) -> Tensor:
        for block in self.gru_blocks:
            x = block(x)
        return x
