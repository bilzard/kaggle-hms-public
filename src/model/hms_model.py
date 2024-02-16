import torch
import torch.nn as nn
from hydra.utils import instantiate

from src.config import ArchitectureConfig


class HmsModel(nn.Module):
    def __init__(
        self,
        cfg: ArchitectureConfig,
        feature_key: str = "eeg",
        pred_key: str = "pred",
        mask_key: str = "cqf",
        pretrained: bool = True,
    ):
        super().__init__()

        self.feature_extractor = instantiate(cfg.model.feature_extractor)
        self.adapters = [instantiate(adapter) for adapter in cfg.model.adapters]
        self.encoder = instantiate(cfg.model.encoder, pretrained=pretrained)
        self.decoder = (
            instantiate(cfg.model.decoder, encoder_channels=self.encoder.out_channels)
            if hasattr(cfg.model, "decoder")
            else None
        )
        self.head = instantiate(
            cfg.model.head,
            in_channels=cfg.model.decoder.hidden_size
            if hasattr(cfg.model, "decoder") and cfg.model.decoder is not None
            else self.encoder.out_channels[-1],
        )
        self.feature_key = feature_key
        self.pred_key = pred_key
        self.mask_key = mask_key

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mask = batch[self.mask_key]
        with torch.autocast(device_type="cuda", enabled=False), torch.no_grad():
            output = self.feature_extractor(batch[self.feature_key], mask)
            spec = output["spectrogram"]
            spec_mask = output["spec_mask"]
            for adapter in self.adapters:
                spec, spec_mask = adapter(spec, spec_mask)

        features = self.encoder(spec)
        if self.decoder is not None:
            x = self.decoder(features)
        else:
            x = features[-1]
        x = self.head(x)

        output = {self.pred_key: x}
        return output


def print_shapes(title: str, data: dict):
    print("-" * 80)
    print(title)
    print("-" * 80)
    for key, value in data.items():
        print(f"{key}: {value.shape}")


def check_model(
    model: HmsModel,
    device="cpu",
    feature_keys=["signal", "channel_mask", "spectrogram", "spec_mask"],
):
    model = model.to(device)
    eeg = torch.randn(2, 2048, 19)
    cqf = torch.randn(2, 2048, 19)

    print_shapes("Input", {"eeg": eeg, "cqf": cqf})
    output = model.feature_extractor(eeg, cqf)
    print_shapes(
        "Feature Extractor", {k: v for k, v in output.items() if k in feature_keys}
    )

    for i, adapter in enumerate(model.adapters):
        output["spectrogram"], output["spec_mask"] = adapter(
            output["spectrogram"], output["spec_mask"]
        )
        print_shapes(
            f"Adapter[{i}]", {k: v for k, v in output.items() if k in feature_keys}
        )

    features = model.encoder(output["spectrogram"])
    print_shapes(
        "Encoder", {f"feature[{i}]": feature for i, feature in enumerate(features)}
    )
    if hasattr(model, "decoder") and model.decoder is not None:
        x = model.decoder(features)
        print_shapes("Decoder", {"pred": x})
    else:
        x = features[-1]

    x = model.head(x)
    print_shapes("Head", {"pred": x})
