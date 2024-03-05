import torch
import torch.nn as nn
from hydra.utils import instantiate
from torch import Tensor

from src.config import ArchitectureConfig


class Head(nn.Module):
    def __init__(self, in_channels: int, bottleneck_ratio: int = 4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // bottleneck_ratio),
            nn.BatchNorm1d(in_channels // bottleneck_ratio),
            nn.PReLU(),
            nn.Linear(in_channels // bottleneck_ratio, 6, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: b c
        return: b 6
        """
        return self.mlp(x)


class HmsModel1d(nn.Module):
    def __init__(
        self,
        cfg: ArchitectureConfig,
        feature_key: str = "eeg",
        pred_key: str = "pred",
        mask_key: str = "cqf",
        spec_key: str = "spec",
        label_key: str = "label",
        weight_key: str = "weight",
        **kwargs,
    ):
        super().__init__()
        self.cfg = cfg
        self.feature_extractor = instantiate(cfg.model.feature_extractor)
        self.eeg_adapter = instantiate(cfg.model.eeg_adapter)
        self.eeg_encoder = instantiate(
            cfg.model.eeg_encoder, in_channels=cfg.in_channels
        )
        self.eeg_feature_processor = instantiate(
            cfg.model.eeg_feature_processor, in_channels=self.eeg_encoder.out_channels
        )
        self.head = Head(in_channels=self.eeg_feature_processor.out_channels)
        self.feature_key = feature_key
        self.pred_key = pred_key
        self.mask_key = mask_key
        self.spec_key = spec_key
        self.label_key = label_key
        self.weight_key = weight_key

    @torch.no_grad()
    def collate_channels(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        eeg = batch[self.feature_key]
        eeg_mask = batch[self.mask_key]

        with torch.autocast(device_type="cuda", enabled=False):
            output = self.feature_extractor(eeg, eeg_mask)

        return output

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            output = self.collate_channels(batch)
            eeg, eeg_mask = output["eeg"], output["eeg_mask"]
            eeg, eeg_mask = self.eeg_adapter(eeg, eeg_mask)
            x = torch.cat([eeg, eeg_mask], dim=1)
        x = self.eeg_encoder(x)
        x = self.eeg_feature_processor(x)
        x = self.head(x)

        output = {self.pred_key: x}
        return output


def print_shapes(title: str, data: dict):
    print("-" * 80)
    print(title)
    print("-" * 80)
    for key, value in data.items():
        print(f"{key}: {value.shape}")


@torch.no_grad()
def check_model(
    model: HmsModel1d,
    device="cpu",
):
    from torchinfo import summary

    model.train()
    model = model.to(device)
    eeg = torch.randn(2, 2048, 19).to(device)
    cqf = torch.randn(2, 2048, 19).to(device)

    print_shapes("Input", {"eeg": eeg, "cqf": cqf})

    output = model.feature_extractor(eeg, cqf)
    print_shapes("Feature Extractor", {k: v for k, v in output.items()})

    eeg = output["eeg"]
    eeg_mask = output["eeg_mask"]

    eeg, eeg_mask = model.eeg_adapter(eeg, eeg_mask)
    print_shapes("Eeg Adapter", dict(eeg=eeg, eeg_mask=eeg_mask))

    x = torch.cat([eeg, eeg_mask], dim=1)
    print_shapes("Merge Mask", {"x": x})

    encoder_input_shape = x.shape
    x = model.eeg_encoder(x)
    print_shapes("Eeg Encoder", {"x": x})

    x = model.eeg_feature_processor(x)
    print_shapes("Eeg Feature Processor", {"x": x})

    x = model.head(x)
    print_shapes("Head", {"x": x})

    print("=" * 80)
    print("Encoder (detail):")
    summary(model.eeg_encoder, input_size=encoder_input_shape)
