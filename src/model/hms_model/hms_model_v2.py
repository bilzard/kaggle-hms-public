import torch
import torch.nn as nn
from einops import rearrange
from hydra.utils import instantiate
from torch import Tensor

from src.config import ArchitectureConfig


class HmsModelV2(nn.Module):
    def __init__(
        self,
        cfg: ArchitectureConfig,
        feature_key: str = "eeg",
        pred_key: str = "pred",
        mask_key: str = "cqf",
        bg_spec_key: str = "bg_spec",
        label_key: str = "label",
        weight_key: str = "weight",
        pretrained: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.feature_extractor = instantiate(cfg.model.feature_extractor)
        self.augmentation = instantiate(cfg.model.augmentation)
        self.adapters = [instantiate(adapter) for adapter in cfg.model.adapters]
        self.bg_adapters = (
            [instantiate(adapter) for adapter in cfg.model.bg_adapters]
            if cfg.use_bg_spec
            else []
        )
        self.spec_transform = (
            instantiate(cfg.model.spec_transform) if cfg.model.spec_transform else None
        )
        self.merger = instantiate(cfg.model.merger) if cfg.use_bg_spec else None
        self.encoder = instantiate(
            cfg.model.encoder,
            pretrained=pretrained,
            in_channels=cfg.in_channels,
        )
        self.decoder = instantiate(
            cfg.model.decoder, encoder_channels=self.encoder.out_channels
        )
        self.feature_processor = instantiate(
            cfg.model.feature_processor, in_channels=self.decoder.output_size
        )
        self.head = instantiate(
            cfg.model.head, in_channels=self.feature_processor.out_channels
        )
        self.feature_key = feature_key
        self.pred_key = pred_key
        self.mask_key = mask_key
        self.bg_spec_key = bg_spec_key
        self.label_key = label_key
        self.weight_key = weight_key

    def generate_spec(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        eeg = batch[self.feature_key]
        eeg_mask = batch[self.mask_key]
        output = self.feature_extractor(eeg, eeg_mask)
        return output

    def compose_spec(
        self, batch: dict[str, Tensor], output: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        spec = output["spec"]
        spec_mask = output["spec_mask"]
        eeg = output["eeg"]
        eeg_mask = output["eeg_mask"]

        if self.cfg.use_bg_spec:
            bg_spec = batch[self.bg_spec_key]
            bg_spec = rearrange(bg_spec, "b f t c -> b c f t")

        if self.training and self.spec_transform is not None:
            spec = self.spec_transform(spec)
            if self.cfg.use_bg_spec:
                bg_spec = self.spec_transform(bg_spec)

        for adapter in self.adapters:
            spec, spec_mask = adapter(spec, spec_mask)
        for bg_adapter in self.bg_adapters:
            bg_spec = bg_adapter(bg_spec)

        if self.cfg.use_bg_spec and bg_spec is not None:
            assert self.merger is not None
            bg_spec_mask = torch.full_like(bg_spec, self.cfg.bg_spec_mask_value).to(
                bg_spec.device
            )
            spec, spec_mask = self.merger(spec, spec_mask, bg_spec, bg_spec_mask)

        output = dict(spec=spec, spec_mask=spec_mask, eeg=eeg, eeg_mask=eeg_mask)

        if self.cfg.input_mask:
            output["spec"] = self.merge_spec_mask(output["spec"], output["spec_mask"])

        return output

    def preprocess(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        output = self.generate_spec(batch)
        if self.training:
            self.augmentation(batch, output)
        output = self.compose_spec(batch, output)
        output.pop("eeg")
        return output

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output = self.preprocess(batch)
        features = self.encoder(output["spec"])
        x = self.decoder(features)
        output = self.feature_processor(dict(spec=x, spec_mask=output["spec_mask"]))
        return output

    def merge_spec_mask(self, spec: Tensor, spec_mask: Tensor) -> Tensor:
        B, C, F, T = spec.shape
        spec_mask = spec_mask.expand(B, C, F, T)
        return torch.cat([spec, spec_mask], dim=1)


def print_shapes(module_name: str, module: nn.Module | None, data: dict):
    print("-" * 80)
    if module is not None:
        print(f"{module_name}: `{module.__class__.__name__}`")
    else:
        print(f"{module_name}")
    print("-" * 80)
    for key, value in data.items():
        print(f"{key}: {value.shape}")


@torch.no_grad()
def check_model(
    model: HmsModelV2,
    device="cpu",
):
    from torchinfo import summary

    model.train()
    model = model.to(device)
    eeg = torch.randn(2, 2048, 19).to(device)
    cqf = torch.randn(2, 2048, 19).to(device)
    bg_spec = torch.randn(2, 4, 100, 256).to(device)

    print_shapes("Input", None, {"eeg": eeg, "cqf": cqf})

    output = model.feature_extractor(eeg, cqf)
    print_shapes(
        "Feature Extractor", model.feature_extractor, {k: v for k, v in output.items()}
    )

    spec = output["spec"]
    spec_mask = output["spec_mask"]

    if model.spec_transform is not None:
        spec = model.spec_transform(spec)
        print_shapes("Spec Transform", model.spec_transform, {"spec": spec})
        if model.cfg.use_bg_spec:
            bg_spec = model.spec_transform(bg_spec)
            print_shapes(
                "Bg Spec Transform", model.spec_transform, {"bg_spec": bg_spec}
            )

    for i, adapter in enumerate(model.adapters):
        spec, spec_mask = adapter(spec, spec_mask)
        print_shapes(
            f"Adapter[{i}]",
            adapter,
            dict(spec=spec, spec_mask=spec_mask),
        )

    for i, bg_adapter in enumerate(model.bg_adapters):
        bg_spec = bg_adapter(bg_spec)
        print_shapes(f"BgAdapter[{i}]", bg_adapter, dict(bg_spec=bg_spec))

    if model.cfg.use_bg_spec:
        assert model.merger is not None
        bg_spec_mask = torch.full_like(bg_spec, model.cfg.bg_spec_mask_value).to(
            bg_spec.device
        )
        spec, spec_mask = model.merger(spec, spec_mask, bg_spec, bg_spec_mask)
        print_shapes("Merger", model.merger, dict(spec=spec, spec_mask=spec_mask))

    if model.cfg.input_mask:
        spec = model.merge_spec_mask(spec, spec_mask)

    encoder_input_shape = spec.shape
    features = model.encoder(spec)
    print_shapes(
        "Encoder",
        model.encoder,
        {f"feature[{i}]": feature for i, feature in enumerate(features)},
    )
    x = model.decoder(features)
    print_shapes("Decoder", model.decoder, {"x": x})

    output = model.feature_processor(dict(spec=x, spec_mask=spec_mask))
    print_shapes("Feature Processor", model.feature_processor, output)

    print("=" * 80)
    print("Encoder (detail):")
    summary(model.encoder, input_size=encoder_input_shape, depth=4)
