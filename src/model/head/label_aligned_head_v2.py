import torch
import torch.nn as nn
from einops import rearrange

from src.model.basic_block import GeMPool2d


class LabelAlignedHeadV2(nn.Module):
    """
    labelに関するドメイン知識を反映したヘッド
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        drop_rate: float = 0.0,
        nonlinearity: bool = True,
        kernel_size: int = 1,
    ):
        super().__init__()
        feat_modules = []
        feat_modules.append(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
                bias=True,
            )
        )
        if nonlinearity:
            feat_modules.append(nn.BatchNorm2d(hidden_channels))
            feat_modules.append(nn.ReLU(inplace=True))

        self.feat = nn.Sequential(*feat_modules)

        self.laterality = nn.Conv2d(
            hidden_channels, 1, kernel_size=1, stride=1, padding="same", bias=True
        )  # -: general, +: lateral
        self.seizure_type = nn.Conv2d(
            hidden_channels, 4, kernel_size=1, stride=1, padding="same", bias=True
        )  # 0: seizure, 1: PD, 2: RDA, 3: others
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

        p_seizure = seizure_type[:, 0]
        p_lpd = seizure_type[:, 1] + laterality
        p_gpd = seizure_type[:, 1] - laterality
        p_lrda = seizure_type[:, 2] + laterality
        p_grda = seizure_type[:, 2] - laterality
        p_other = seizure_type[:, 3]

        x = torch.stack([p_seizure, p_lpd, p_gpd, p_lrda, p_grda, p_other], dim=1)
        x = rearrange(x, "b c 1 1 -> b c")

        return x
