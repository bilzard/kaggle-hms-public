import torch.nn as nn
from einops import rearrange
from torch import Tensor

from src.model.basic_block.util import calc_similarity


class CosineSimilarityEncoder3d(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        activation: nn.Module,
        channel_dim: int = 1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.channel_dim = channel_dim
        self.similarity_encoder = nn.Sequential(
            nn.Conv3d(1, self.hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm3d(self.hidden_dim),
            activation,
        )

    def forward(self, x_left: Tensor, x_right: Tensor) -> Tensor:
        """
        x_left: b c ch f t
        x_right: b c ch f t
        output: b c ch f t
        """
        assert x_left.shape == x_right.shape, f"{x_left.shape} != {x_right.shape}"
        assert len(x_left.shape) == 5, f"dimension should be 5, got {x_left.shape}"

        sim = calc_similarity(x_left, x_right)
        sim = self.similarity_encoder(sim)
        return sim


class CosineSimilarityEncoder2d(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        activation: nn.Module,
        channel_dim: int = 1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.channel_dim = channel_dim
        self.similarity_encoder = nn.Sequential(
            nn.Conv2d(1, self.hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.hidden_dim),
            activation,
        )

    def forward(self, x_left: Tensor, x_right: Tensor) -> Tensor:
        """
        x_left: b c f t
        x_right: b c f t
        output: b c f t
        """
        assert x_left.shape == x_right.shape, f"{x_left.shape} != {x_right.shape}"
        assert len(x_left.shape) == 4, f"dimension should be 4, got {x_left.shape}"

        sim = calc_similarity(x_left, x_right)
        sim = self.similarity_encoder(sim)
        return sim


class CosineSimilarityEncoder1d(CosineSimilarityEncoder2d):
    def __init__(
        self,
        hidden_dim: int,
        activation: nn.Module,
        channel_dim: int = 1,
    ):
        super().__init__(
            hidden_dim=hidden_dim, channel_dim=channel_dim, activation=activation
        )

    def forward(self, x_left: Tensor, x_right: Tensor) -> Tensor:
        """
        x_left: b c t
        x_right: b c t
        output: b c t
        """
        assert x_left.shape == x_right.shape, f"{x_left.shape} != {x_right.shape}"
        assert len(x_left.shape) == 3, f"dimension should be 3, got {x_left.shape}"

        x_left = rearrange(x_left, "b c t -> b c 1 t")
        x_right = rearrange(x_right, "b c t -> b c 1 t")
        sim = super().forward(x_left, x_right)
        sim = rearrange(sim, "b c 1 t -> b c t")

        return sim


if __name__ == "__main__":
    import torch
    from torchinfo import summary

    x_left = torch.randn(2, 3, 4, 5)
    x_right = torch.randn(2, 3, 4, 5)
    hidden_dim = 8
    model = CosineSimilarityEncoder2d(hidden_dim, nn.PReLU())
    sim = model(x_left, x_right)
    B, C, F, T = x_left.shape
    assert sim.shape == (B, hidden_dim, F, T), f"x_left={x_left.shape}, sim={sim.shape}"
    summary(model, input_data=(x_left, x_right))

    x_left = torch.randn(2, 3, 5)
    x_right = torch.randn(2, 3, 5)
    hidden_dim = 8
    model = CosineSimilarityEncoder1d(hidden_dim, nn.PReLU())
    sim = model(x_left, x_right)
    B, C, T = x_left.shape
    assert sim.shape == (B, hidden_dim, T), f"x_left={x_left.shape}, sim={sim.shape}"
    summary(model, input_data=(x_left, x_right))
