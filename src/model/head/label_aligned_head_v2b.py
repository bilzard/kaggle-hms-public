import torch
import torch.nn as nn
from einops import rearrange

from src.model.basic_block import ClampedSigmoid, ConvBnPReLu2d, GeMPool2d


class LabelAlignedHeadV2b(nn.Module):
    """
    labelに関するドメイン知識を反映したヘッド

    changes from V2:
    - 足し算だとlateralityの学習がうまく進まない可能性があるので掛け算にする
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        drop_rate: float = 0.0,
        kernel_size: int = 1,
        delta: float = 1e-2,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.eps = eps

        self.feat = nn.Sequential(
            ConvBnPReLu2d(in_channels, hidden_channels, kernel_size=1),
            ConvBnPReLu2d(
                hidden_channels,
                hidden_channels,
                kernel_size=kernel_size,
                groups=hidden_channels,
            ),
            ConvBnPReLu2d(hidden_channels, hidden_channels, kernel_size=1),
        )
        self.laterality = nn.Sequential(
            ConvBnPReLu2d(hidden_channels, hidden_channels, kernel_size=1, stride=1),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, stride=1, bias=True),
            ClampedSigmoid(delta),
        )
        self.seizure_type = nn.Sequential(
            ConvBnPReLu2d(hidden_channels, hidden_channels, kernel_size=1, stride=1),
            nn.Conv2d(hidden_channels, 4, kernel_size=1, stride=1, bias=False),
            ClampedSigmoid(delta),
        )
        self.pool = GeMPool2d()
        self.drop_rate = drop_rate
        self.dropout = nn.Dropout2d(drop_rate, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, in_channels, h, w)
        return: (batch_size, out_channels)
        """
        if self.drop_rate > 0:
            x = self.dropout(x)
        x = self.feat(x)
        x = self.pool(x)

        laterality = self.laterality(x).squeeze(dim=1)
        seizure_type = self.seizure_type(x)

        p_seizure = torch.logit(seizure_type[:, 0], eps=self.eps)
        p_lpd = torch.logit(seizure_type[:, 1] * laterality, eps=self.eps)
        p_gpd = torch.logit(seizure_type[:, 1] * (1 - laterality), eps=self.eps)
        p_lrda = torch.logit(seizure_type[:, 2] * laterality, eps=self.eps)
        p_grda = torch.logit(seizure_type[:, 2] * (1 - laterality), eps=self.eps)
        p_other = torch.logit(seizure_type[:, 3], eps=self.eps)

        x = torch.stack([p_seizure, p_lpd, p_gpd, p_lrda, p_grda, p_other], dim=1)
        x = rearrange(x, "b c 1 1 -> b c")

        return x
