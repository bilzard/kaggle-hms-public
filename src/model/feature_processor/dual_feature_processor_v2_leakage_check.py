import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from src.model.basic_block import (
    CosineSimilarityEncoder2d,
    GeMPool2d,
    vector_pair_mapping,
)
from src.model.feature_processor.base import BaseFeatureProcessor


class DualFeatureProcessorV2LeakageCheck(BaseFeatureProcessor):
    """
    w < 0.3 or not特徴(is_clean特徴)をモデルに明示的に与えてCVスコアにどう寄与するかを調べる。
    予想としては、trainに2つのデータソースがあり、それぞれラベルの平均分布が異なるとした場合、
    is_clean特徴を与えたモデルは、与えないモデルよりもCVを過小に下げることが可能になると考えられる。
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        activation: nn.Module,
        lr_mapping_type: str = "identity",
        binary_weight: bool = True,
    ):
        super().__init__(in_channels=in_channels)
        self.hidden_dim = hidden_dim
        self.lr_mapping_type = lr_mapping_type
        self.binary_weight = binary_weight

        self.similarity_encoder = CosineSimilarityEncoder2d(
            hidden_dim=hidden_dim, activation=activation
        )
        self.weight_embedding = nn.Linear(1, hidden_dim)
        self.pool = GeMPool2d()

    @property
    def out_channels(self) -> int:
        return 2 * self.in_channels + 2 * self.hidden_dim

    def forward(self, inputs: dict[str, Tensor]) -> Tensor:
        """
        inputs:
        - spec: (d b) c f t
        output: b c
        """
        x = inputs["spec"]
        x = rearrange(x, "(d b) c f t -> d b c f t", d=2)
        x_left, x_right = x[0], x[1]  # b c f t
        feats = []

        feats.extend(list(vector_pair_mapping(x_left, x_right, self.lr_mapping_type)))

        sim = self.similarity_encoder(x_left, x_right)
        feats.append(sim)

        x = torch.cat(feats, dim=1)

        x = self.pool(x)  # b c 1 1
        x = rearrange(x, "b c 1 1 -> b c")

        weight = inputs["weight"]  # b 1

        if self.binary_weight:
            weight = (weight < 0.3).float().to(weight.device)

        weight_embed = self.weight_embedding(weight)  # b h
        x = torch.cat([x, weight_embed], dim=1)

        return x


if __name__ == "__main__":
    from torchinfo import summary

    duality = 2
    batch_size = 2
    in_channels = 128
    hidden_dim = 64
    F, T = 16, 16
    feature_processor = DualFeatureProcessorV2LeakageCheck(
        in_channels, hidden_dim, nn.PReLU()
    )
    spec = torch.randn(duality * batch_size, in_channels, F, T)
    weight = torch.randn(batch_size, 1)
    inputs = dict(spec=spec, weight=weight)
    output = feature_processor(inputs)
    print(f"{output.shape=}")
    assert output.shape == (2, 2 * in_channels + 2 * hidden_dim), f"{output.shape=}"

    summary(feature_processor, input=inputs)
