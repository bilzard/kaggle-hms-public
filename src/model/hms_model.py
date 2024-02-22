import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from hydra.utils import instantiate
from torch import Tensor

from src.config import ArchitectureConfig


def calc_similarity(x: Tensor, y: Tensor, channel_dim: int = 1, eps=1e-4) -> Tensor:
    """
    x: (B, C, F, T)
    y: (B, C, F, T)

    Returns:
    similarity: (B, 1, F, T)
    """

    return F.cosine_similarity(x, y, dim=channel_dim, eps=eps).unsqueeze(channel_dim)


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
        self.cfg = cfg

        self.is_dual = cfg.is_dual

        self.sample_collator = instantiate(cfg.model.sample_collator)
        self.feature_extractor = instantiate(cfg.model.feature_extractor)
        self.adapters = [instantiate(adapter) for adapter in cfg.model.adapters]

        self.encoder = instantiate(
            cfg.model.encoder,
            pretrained=pretrained,
            in_channels=cfg.in_channels,
        )
        self.decoder = instantiate(
            cfg.model.decoder, encoder_channels=self.encoder.out_channels
        )

        match cfg.merge_type, cfg.map_similarity:
            case "cat", _:
                similarity_dim = cfg.hidden_dim if cfg.map_similarity else 1
                agg_input_channel_size = (
                    2 * self.decoder.output_size + similarity_dim
                    if self.is_dual
                    else self.decoder.output_size
                )
            case "add", True:
                similarity_dim = self.decoder.output_size
                agg_input_channel_size = (
                    similarity_dim if self.is_dual else self.decoder.output_size
                )
            case _:
                raise ValueError(f"Invalid merge_type: {self.merge_type}")

        if cfg.map_similarity:
            self.similarity_encoder = nn.Sequential(
                nn.Conv2d(1, similarity_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(similarity_dim),
                nn.PReLU(),
            )

        self.sample_aggregator = instantiate(
            cfg.model.sample_aggregator,
            input_channels=agg_input_channel_size,
        )
        self.head = instantiate(
            cfg.model.head, in_channels=self.sample_aggregator.output_size
        )
        self.feature_key = feature_key
        self.pred_key = pred_key
        self.mask_key = mask_key

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mask = batch[self.mask_key]
        feature = batch[self.feature_key]

        if len(feature.shape) == 3:
            feature = feature.unsqueeze(1)
            mask = mask.unsqueeze(1)

        B, S, T, C = feature.shape
        with torch.autocast(device_type="cuda", enabled=False), torch.no_grad():
            feature, mask = self.sample_collator(feature, mask)
            output = self.feature_extractor(feature, mask)
            spec = output["spectrogram"]
            spec_mask = output["spec_mask"]
            for adapter in self.adapters:
                spec, spec_mask = adapter(spec, spec_mask)

        features = self.encoder(spec)
        x = self.decoder(features)
        if self.is_dual:
            x = self.recover_dual(x)

        x = self.sample_aggregator(x, num_samples=S)
        x = self.head(x)

        output = {self.pred_key: x}
        return output

    def recover_dual(self, x: Tensor) -> Tensor:
        x = rearrange(x, "(d b) c f t -> d b c f t", d=2)
        x_left = x[0]
        x_right = x[1]
        sim = calc_similarity(x_left, x_right)

        if self.cfg.map_similarity:
            sim = self.similarity_encoder(sim)

        match self.cfg.merge_type:
            case "add":
                x = x_left + x_right + sim
            case "cat":
                x = torch.cat([x_left, x_right, sim], dim=1)
            case _:
                raise ValueError(f"Invalid merge_type: {self.merge_type}")
        return x


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
    eeg = torch.randn(2, 2, 2048, 19)
    cqf = torch.randn(2, 2, 2048, 19)

    B, S, T, C = eeg.shape

    print_shapes("Input", {"eeg": eeg, "cqf": cqf})

    cqf, eeg = model.sample_collator(cqf, eeg)
    print_shapes("SampleCollator", {"eeg": eeg, "cqf": cqf})

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
    x = model.decoder(features)
    print_shapes("Decoder", {"x": x})

    if model.is_dual:
        x = model.recover_dual(x)
        print_shapes("recover dual", {"x": x})

    x = model.sample_aggregator(x, num_samples=S)
    print_shapes("SampleAggregator", {"x": x})

    x = model.head(x)
    print_shapes("Head", {"x": x})
