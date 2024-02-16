import torch
import torch.nn as nn


class LabelAlignedHeadV3(nn.Module):
    """
    labelに関するドメイン知識を反映したヘッド

    decoderでattentionを行っている前提
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        drop_rate: float = 0.0,
        nonlinearity: bool = True,
    ):
        super().__init__()
        feat_modules = []
        feat_modules.append(
            nn.Linear(
                in_channels,
                hidden_channels,
                bias=True,
            )
        )
        if nonlinearity:
            feat_modules.append(nn.BatchNorm1d(hidden_channels))
            feat_modules.append(nn.ReLU(inplace=True))

        self.feat = nn.Sequential(*feat_modules)

        self.laterality = nn.Linear(
            hidden_channels, 1, bias=True
        )  # -: general, +: lateral
        self.seizure_type = nn.Linear(
            hidden_channels, 4, bias=True
        )  # 0: seizure, 1: PD, 2: RDA, 3: others
        self.drop_rate = drop_rate
        self.dropout = nn.Dropout(drop_rate, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, in_channels, freq, time)
        return: (batch_size, out_channels)
        """
        x = x.sum(dim=(2, 3))

        if self.drop_rate > 0:
            x = self.dropout(x)
        x = self.feat(x)

        laterality = self.laterality(x).squeeze(dim=1)
        seizure_type = self.seizure_type(x)

        p_seizure = seizure_type[:, 0]
        p_lpd = seizure_type[:, 1] + laterality
        p_gpd = seizure_type[:, 1] - laterality
        p_lrda = seizure_type[:, 2] + laterality
        p_grda = seizure_type[:, 2] - laterality
        p_other = seizure_type[:, 3]

        x = torch.stack([p_seizure, p_lpd, p_gpd, p_lrda, p_grda, p_other], dim=1)

        return x
