import torch.nn as nn
from torch import Tensor


class GruBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        n_layers=1,
        bidirectional: bool = True,
        bottleneck_ratio: int = 4,
        use_ff: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.use_ff = use_ff

        self.norm = nn.LayerNorm(hidden_size)
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            n_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        directional_factor = 2 if bidirectional else 1
        self.map = nn.Sequential(
            nn.Linear(hidden_size * directional_factor, hidden_size),
            nn.GELU() if use_ff else nn.Identity(),
        )
        if use_ff:
            self.ff = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size // bottleneck_ratio),
                nn.GELU(),
                nn.Linear(hidden_size // bottleneck_ratio, hidden_size),
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
