import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from src.model.basic_block import (
    ConvBnAct2d,
    CosineSimilarityEncoder2d,
    GeMPool2d,
    Mlp,
    vector_pair_mapping,
)
from src.model.feature_processor.base import BaseFeatureProcessor


class SqueezeExcite(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        se_ratio: int,
        activation: type[nn.Module],
    ):
        super().__init__()

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(hidden_dim, hidden_dim // se_ratio, kernel_size=1),
            activation(),
            nn.Conv2d(hidden_dim // se_ratio, hidden_dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.se(x)


class Pool(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = GeMPool2d()

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(x)
        x = rearrange(x, "b c 1 1 -> b c")
        return x


class InvertedResidual(nn.Module):
    def __init__(
        self,
        depth_multiplier: int,
        hidden_dim: int,
        kernel_size: int,
        se_ratio: int,
        activation: type[nn.Module],
        has_skip: bool = True,
    ):
        super().__init__()
        self.has_skip = has_skip

        self.ir_conv = nn.Sequential(
            ConvBnAct2d(
                hidden_dim,
                hidden_dim * depth_multiplier,
                activation=activation,
            ),
            ConvBnAct2d(
                hidden_dim * depth_multiplier,
                hidden_dim * depth_multiplier,
                kernel_size=(kernel_size, kernel_size),
                activation=activation,
                groups=hidden_dim,
            ),
            SqueezeExcite(
                hidden_dim * depth_multiplier, se_ratio=se_ratio, activation=activation
            ),
            ConvBnAct2d(
                hidden_dim * depth_multiplier,
                hidden_dim,
                activation=activation,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.has_skip:
            return x + self.ir_conv(x)

        return self.ir_conv(x)


class ConvFeat(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        kernel_size: int,
        activation: type[nn.Module],
        n_layers: int = 1,
        depth_multiplier: int = 4,
        se_ratio: int = 4,
        use_ir_conv: bool = True,
        has_skip: bool = True,
        use_se: bool = True,
    ):
        super().__init__()
        self.has_skip = has_skip

        self.conv = nn.Sequential(
            ConvBnAct2d(
                in_channels,
                hidden_dim,
                kernel_size=(1, 1),
                activation=activation,
            ),
            *[
                InvertedResidual(
                    depth_multiplier=depth_multiplier,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    se_ratio=se_ratio,
                    activation=activation,
                )
                if use_ir_conv
                else ConvBnAct2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=(kernel_size, kernel_size),
                    activation=activation,
                )
                for _ in range(n_layers)
            ],
            SqueezeExcite(hidden_dim, se_ratio=se_ratio, activation=activation)
            if use_se
            else nn.Identity(),
        )
        self.pool = GeMPool2d()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = rearrange(x, "b c 1 1 -> b c")
        return x


class DualFeatureProcessorWithAuxHead(BaseFeatureProcessor):
    """
    weightの回帰headを持つDualFeatureProcessor
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        activation: nn.Module,
        lr_mapping_type: str = "identity",
        num_heads: int = 1,
        bottleneck_ratio: int = 4,
        kernel_size: int = 3,
        n_layers: int = 1,
        only_pool: bool = True,
        use_weight_embedding: bool = False,
        depth_multiplier: int = 4,
        se_ratio: int = 4,
        use_ir_conv: bool = True,
        use_se: bool = True,
        decouple_two_branches: bool = False,
    ):
        super().__init__(in_channels=in_channels)
        self.hidden_dim = hidden_dim
        self.lr_mapping_type = lr_mapping_type
        self.num_heads = num_heads
        self.use_weight_embedding = use_weight_embedding
        self._frozen_aux_branch = False
        self.decouple_two_branches = decouple_two_branches

        self.similarity_encoder = CosineSimilarityEncoder2d(
            hidden_dim=hidden_dim, activation=activation
        )

        num_channels = self.out_channels
        num_channels_aux = self.out_channels
        if only_pool:
            self.conv = Pool()
            self.aux_conv = Pool()
        else:
            self.conv = ConvFeat(
                in_channels=num_channels,
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                n_layers=n_layers,
                depth_multiplier=depth_multiplier,
                se_ratio=se_ratio,
                use_ir_conv=use_ir_conv,
                activation=nn.PReLU,
                use_se=use_se,
            )
            self.aux_conv = ConvFeat(
                in_channels=num_channels_aux,
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                n_layers=n_layers,
                depth_multiplier=depth_multiplier,
                se_ratio=se_ratio,
                use_ir_conv=use_ir_conv,
                activation=nn.PReLU,
                use_se=use_se,
            )
            num_channels = hidden_dim
            num_channels_aux = hidden_dim

        if use_weight_embedding:
            num_channels += hidden_dim

        self.head = Mlp(
            in_channels=num_channels,
            out_channels=6 * num_heads,
            bottleneck_ratio=bottleneck_ratio,
        )
        self.aux_head = Mlp(
            in_channels=num_channels_aux,
            out_channels=1 * num_heads,
            bottleneck_ratio=bottleneck_ratio,
        )
        self.weight_embedding = nn.Linear(1, hidden_dim)

    @property
    def out_channels(self) -> int:
        return 2 * self.in_channels + self.hidden_dim

    def freeze_aux_branch(self):
        """
        w>=0.3に絞ったあとはweightの分布が正例のみになる。
        偏った分布で学習すると性能が劣化するのでw>=0に絞り始めたepochでaux_branchをfreezeできるようにする。
        """
        if not self._frozen_aux_branch:
            print(f"* {self.__class__.__name__}: freeze aux branch")
            self.aux_conv.requires_grad_(False)
            self.aux_head.requires_grad_(False)
            self._frozen_aux_branch = True

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        inputs:
        - spec: (d b) c f t
        output:
        - pred: b h c
        - weight: b h
        """
        x = inputs["spec"]
        x = rearrange(x, "(d b) c f t -> d b c f t", d=2)
        x_left, x_right = x[0], x[1]  # b c f t
        feats = []

        feats.extend(list(vector_pair_mapping(x_left, x_right, self.lr_mapping_type)))

        sim = self.similarity_encoder(x_left, x_right)
        feats.append(sim)

        feat = torch.cat(feats, dim=1)  # b c f t

        logit = self.conv(feat)  # b c
        weight = self.aux_conv(feat)  # b c

        weight = self.aux_head(weight)  # b h

        if self.use_weight_embedding:
            weight_emb = weight[:, 0]  # b
            if self.decouple_two_branches:
                weight_emb = weight_emb.detach()
            weight_emb = rearrange(weight_emb, "b -> b 1")  # b 1
            weight_emb = weight_emb.sigmoid()
            weight_emb = self.weight_embedding(weight_emb)  # b c
            logit = torch.cat([logit, weight_emb], dim=1)  # b c

        logit = self.head(logit)  # b (h 6)
        logit = rearrange(logit, "b (h c) -> b h c", h=self.num_heads)  # b h c

        output = dict(pred=logit, weight=weight)
        return output


if __name__ == "__main__":
    from torchinfo import summary

    def print_shapes(module_name: str, module: nn.Module | None, data: dict):
        print("-" * 80)
        if module is not None:
            print(f"{module_name}: `{module.__class__.__name__}`")
        else:
            print(f"{module_name}")
        print("-" * 80)
        for key, value in data.items():
            print(f"{key}: {value.shape}")

    duality = 2
    batch_size = 2
    in_channels = 128
    hidden_dim = 32
    F, T = 16, 16
    n_layers = 1
    feature_processor = DualFeatureProcessorWithAuxHead(
        in_channels, hidden_dim, nn.PReLU(), num_heads=2
    )
    spec = torch.randn(duality * batch_size, in_channels, F, T)
    inputs = dict(spec=spec)
    output = feature_processor(inputs)
    print_shapes("inputs", None, inputs)
    print_shapes("output", None, output)

    summary(feature_processor, input=inputs)
