import torch
import torch.nn as nn
from einops import rearrange

from src.model.basic_block import GeMPool2d, InverseSoftmax


class LabelAlignedHead(nn.Module):
    """
    labelに関するドメイン知識を反映したヘッド
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        drop_rate: float = 0.0,
        eps: float = 1e-4,
        C: float = 3.0,
    ):
        super().__init__()
        self.feat = nn.Conv2d(
            in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.laterality = nn.Conv2d(
            hidden_channels, 1, kernel_size=1, stride=1, padding=0, bias=True
        )  # 0: general, 1: lateral
        self.seizure_type = nn.Conv2d(
            hidden_channels, 4, kernel_size=1, stride=1, padding=0, bias=True
        )  # 0: seizure, 1: PD, 2: RDA, 3: others
        self.pool = GeMPool2d()
        self.drop_rate = drop_rate
        self.dropout = nn.Dropout2d(drop_rate, inplace=True)

        self.inverse_softmax = InverseSoftmax(dim=1, eps=eps, C=C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, in_channels, h, w)
        return: (batch_size, out_channels)
        """
        if self.drop_rate > 0:
            x = self.dropout(x)
        x = self.feat(x)
        x = self.pool(x)

        laterality = self.laterality(x).sigmoid().squeeze(dim=1)
        seizure_type = self.seizure_type(x).softmax(dim=1)

        p_seizure = seizure_type[:, 0]
        p_lpd = seizure_type[:, 1] * laterality
        p_gpd = seizure_type[:, 1] * (1 - laterality)
        p_lrda = seizure_type[:, 2] * laterality
        p_grda = seizure_type[:, 2] * (1 - laterality)
        p_other = seizure_type[:, 3]

        x = torch.stack([p_seizure, p_lpd, p_gpd, p_lrda, p_grda, p_other], dim=1)
        x = self.inverse_softmax(x)

        x = rearrange(x, "b c 1 1 -> b c")

        return x
