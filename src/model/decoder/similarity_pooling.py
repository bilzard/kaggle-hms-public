import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from src.model.basic_block import ConvBnPReLu2d, calc_similarity


class SimilarityPooling(nn.Module):
    def __init__(
        self,
        encoder_channels: list[int],
        hidden_dim: int = 64,
        depth: int = 4,
        debug: bool = False,
    ):
        super().__init__()
        self.channels = encoder_channels[-depth:]
        self.depth = depth
        self.debug = debug

        self.mappers = nn.ModuleList(
            [ConvBnPReLu2d(ch, hidden_dim, kernel_size=1) for ch in self.channels]
        )
        self.sim_encoder = ConvBnPReLu2d(depth, hidden_dim, kernel_size=1)
        self.sim_pools = nn.ModuleList(
            [
                nn.MaxPool2d(2 ** (depth - i - 1), 2 ** (depth - i - 1))
                for i in range(depth)
            ]
        )
        self._output_size = hidden_dim * 3

    @property
    def output_size(self) -> int:
        return self._output_size

    def forward(self, features: list[Tensor]) -> Tensor:
        """
        input
        -----
        features: list of Tensor
            list of feature maps from the encoder

        output
        ------
        x: Tensor
            last feature map + similarity pooled feature map
        """
        features = features[-self.depth :]
        for i, (feat, mapper) in enumerate(zip(features, self.mappers)):
            features[i] = mapper(feat)

        last_feat = features[-1]
        _, _, h, w = last_feat.size()
        for i, (feat, pool) in enumerate(zip(features, self.sim_pools)):
            feat = rearrange(feat, "(d b) c f t -> d b c f t", d=2)
            feat = calc_similarity(feat[0], feat[1])
            feat = pool(feat)
            features[i] = feat

        sim_feat = torch.cat(features, dim=1)  # (B, C, F, T)
        sim_feat = self.sim_encoder(sim_feat)

        output = rearrange(last_feat, "(d b) c f t -> d b c f t", d=2)
        if self.debug:
            print("last_feat:", tuple(last_feat.shape))
            print("sim features:", tuple(sim_feat.shape))
        output = torch.cat([output[0], output[1], sim_feat], dim=1)

        return output


if __name__ == "__main__":
    import torch
    import torch.nn as nn

    encoder_channels = [10, 20, 40, 80, 160, 320]
    features = []
    for i in range(len(encoder_channels), 0, -1):
        x = torch.randn(2, encoder_channels[-i], 10 * 2**i, 10 * 2**i)
        features.append(x)

    for i in range(len(features)):
        print(f"features[{i}]:", tuple(features[i].shape))

    model = SimilarityPooling(encoder_channels, debug=True)
    output = model(features)
    print("output:", tuple(output.shape))
