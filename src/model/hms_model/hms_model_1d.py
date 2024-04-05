import torch
import torch.nn as nn
from hydra.utils import instantiate
from torch import Tensor

from src.config import ArchitectureConfig


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
        self.augmentation = instantiate(cfg.model.augmentation)
        self.eeg_pre_adapter = instantiate(cfg.model.eeg_pre_adapter)
        self.eeg_adapter = instantiate(cfg.model.eeg_adapter)
        self.eeg_encoder = instantiate(
            cfg.model.eeg_encoder, in_channels=cfg.in_channels
        )
        self.eeg_feature_processor = instantiate(
            cfg.model.eeg_feature_processor, in_channels=self.eeg_encoder.out_channels
        )
        self.head = instantiate(
            cfg.model.head, in_channels=self.eeg_feature_processor.out_channels
        )
        self.post_adapter = instantiate(cfg.model.post_adapter)
        self.feature_key = feature_key
        self.pred_key = pred_key
        self.mask_key = mask_key
        self.spec_key = spec_key
        self.label_key = label_key
        self.weight_key = weight_key

    def preprocess(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        eeg = batch[self.feature_key]
        eeg_mask = batch[self.mask_key]

        output = self.feature_extractor(eeg, eeg_mask)
        output["eeg"], output["eeg_mask"] = self.eeg_pre_adapter(
            output["eeg"], output["eeg_mask"]
        )
        if self.training:
            self.augmentation(batch, output)

        eeg, eeg_mask = output["eeg"], output["eeg_mask"]
        eeg, eeg_mask = self.eeg_adapter(eeg, eeg_mask)

        if self.cfg.input_mask:
            output["eeg"] = torch.cat([eeg, eeg_mask], dim=1)
        else:
            output["eeg"] = eeg

        return output

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output = self.preprocess(batch)
        x = self.eeg_encoder(output["eeg"])
        x = self.eeg_feature_processor(x)
        x = self.head(x)
        if not self.training:
            x = self.post_adapter(x)

        output = {self.pred_key: x}
        return output


def print_shapes(module_name: str, module: nn.Module | None, data: dict):
    print("-" * 80)
    if module is not None:
        print(f"{module_name}: `{module.__class__.__name__}`")
    else:
        print(f"{module_name}")
    print("-" * 80)
    for key, value in data.items():
        if isinstance(value, Tensor):
            print(f"{key}: {value.shape}")


@torch.no_grad()
def check_model(
    model: HmsModel1d,
    device="cpu",
):
    from torchinfo import summary

    model.train()
    model = model.to(device)
    eeg = torch.randn(2, 10240, 19).to(device)
    cqf = torch.randn(2, 10240, 19).to(device)
    weight = torch.randn(2, 1).to(device)
    label = torch.randn(2, 6).to(device)
    batch = dict(eeg=eeg, cqf=cqf, weight=weight, label=label)

    print_shapes("Input", None, batch)

    output = model.feature_extractor(batch["eeg"], batch["cqf"])
    print_shapes(
        "Feature Extractor", model.feature_extractor, {k: v for k, v in output.items()}
    )

    output["eeg"], output["eeg_mask"] = model.eeg_pre_adapter(
        output["eeg"], output["eeg_mask"]
    )
    print_shapes("Eeg Pre Adapter", model.eeg_pre_adapter, output)

    model.augmentation(batch, output)
    print_shapes("Augmentation", model.augmentation, output)

    eeg = output["eeg"]
    eeg_mask = output["eeg_mask"]

    eeg, eeg_mask = model.eeg_adapter(eeg, eeg_mask)
    print_shapes("Eeg Adapter", model.eeg_adapter, dict(eeg=eeg, eeg_mask=eeg_mask))

    if model.cfg.input_mask:
        x = torch.cat([eeg, eeg_mask], dim=1)
        print_shapes("Merge Mask", None, {"x": x})
    else:
        x = eeg

    encoder_input_shape = x.shape
    x = model.eeg_encoder(x)
    print_shapes("Eeg Encoder", model.eeg_encoder, {"x": x})

    x = model.eeg_feature_processor(x)
    print_shapes("Eeg Feature Processor", model.eeg_feature_processor, {"x": x})

    x = model.head(x)
    print_shapes("Head", model.head, {"x": x})

    x = model.post_adapter(x)
    print_shapes("Post Adapter", model.post_adapter, {"x": x})

    print("=" * 80)
    print("Encoder (detail):")
    summary(model.eeg_encoder, input_size=encoder_input_shape, depth=5)
