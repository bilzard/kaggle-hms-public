import torch
import torch.nn as nn
from einops import rearrange
from hydra.utils import instantiate
from torch import Tensor

from src.config import ArchitectureConfig
from src.model.basic_block import calc_similarity, vector_pair_mapping


class HmsModel(nn.Module):
    def __init__(
        self,
        cfg: ArchitectureConfig,
        feature_key: str = "eeg",
        pred_key: str = "pred",
        mask_key: str = "cqf",
        spec_key: str = "spec",
        pretrained: bool = True,
    ):
        super().__init__()
        self.cfg = cfg

        self.recover_dual = cfg.recover_dual

        self.feature_extractor = instantiate(cfg.model.feature_extractor)
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

        similarity_dim = cfg.hidden_dim if cfg.use_similarity_feature else 0
        head_input_channel_size = (
            2 * self.decoder.output_size + similarity_dim
            if self.recover_dual
            else self.decoder.output_size
        )

        if cfg.use_similarity_feature:
            self.similarity_encoder = nn.Sequential(
                nn.Conv2d(1, similarity_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(similarity_dim),
                nn.PReLU(),
            )

        self.head = instantiate(cfg.model.head, in_channels=head_input_channel_size)
        self.feature_key = feature_key
        self.pred_key = pred_key
        self.mask_key = mask_key
        self.spec_key = spec_key

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mask = batch[self.mask_key]
        feature = batch[self.feature_key]
        if self.cfg.use_bg_spec:
            bg_spec = batch[self.spec_key]
            bg_spec = rearrange(bg_spec, "b f t c -> b c f t")

        with torch.autocast(device_type="cuda", enabled=False), torch.no_grad():
            output = self.feature_extractor(feature, mask)
            spec = output["spectrogram"]
            spec_mask = output["spec_mask"]

            if self.training and self.spec_transform is not None:
                spec = self.spec_transform(spec)
                if self.cfg.use_bg_spec:
                    bg_spec = self.spec_transform(bg_spec)

            for adapter in self.adapters:
                spec, spec_mask = adapter(spec, spec_mask)
            for bg_adapter in self.bg_adapters:
                bg_spec = bg_adapter(bg_spec)

            if self.cfg.use_bg_spec:
                assert self.merger is not None
                bg_spec_mask = torch.ones_like(bg_spec).to(bg_spec.device)
                spec, spec_mask = self.merger(spec, spec_mask, bg_spec, bg_spec_mask)

        if self.cfg.input_mask:
            spec = self.merge_spec_mask(spec, spec_mask)

        features = self.encoder(spec)
        x = self.decoder(features)
        if self.recover_dual:
            x = self._do_recover_dual(x)

        x = self.head(x)

        output = {self.pred_key: x}
        return output

    def _do_recover_dual(self, x: Tensor) -> Tensor:
        x = rearrange(x, "(d b) c f t -> d b c f t", d=2)
        x_left = x[0]
        x_right = x[1]

        if self.cfg.use_similarity_feature:
            sim = calc_similarity(x_left, x_right)
            sim = self.similarity_encoder(sim)
            x_left, x_right = vector_pair_mapping(
                x_left, x_right, self.cfg.lr_mapping_type
            )
            x = torch.cat([x_left, x_right, sim], dim=1)
        else:
            x_left, x_right = vector_pair_mapping(
                x_left, x_right, self.cfg.lr_mapping_type
            )
            x = torch.cat([x_left, x_right], dim=1)
        return x

    def merge_spec_mask(self, spec: Tensor, spec_mask: Tensor) -> Tensor:
        B, C, F, T = spec.shape
        spec_mask = spec_mask.expand(B, C, F, T)
        return torch.cat([spec, spec_mask], dim=1)


def print_shapes(title: str, data: dict):
    print("-" * 80)
    print(title)
    print("-" * 80)
    for key, value in data.items():
        print(f"{key}: {value.shape}")


@torch.no_grad()
def check_model(
    model: HmsModel,
    device="cpu",
    feature_keys=["signal", "channel_mask", "spectrogram", "spec_mask"],
):
    model.train()
    model = model.to(device)
    eeg = torch.randn(2, 2048, 19).to(device)
    cqf = torch.randn(2, 2048, 19).to(device)
    bg_spec = torch.randn(2, 4, 100, 256).to(device)

    print_shapes("Input", {"eeg": eeg, "cqf": cqf})

    output = model.feature_extractor(eeg, cqf)
    print_shapes(
        "Feature Extractor", {k: v for k, v in output.items() if k in feature_keys}
    )

    spec = output["spectrogram"]
    spec_mask = output["spec_mask"]

    if model.spec_transform is not None:
        spec = model.spec_transform(spec)
        print_shapes(f"Spec Transform - {model.spec_transform}", {"spec": spec})
        if model.cfg.use_bg_spec:
            bg_spec = model.spec_transform(bg_spec)
            print_shapes(
                f"Bg Spec Transform - {model.spec_transform}", {"bg_spec": bg_spec}
            )

    for i, adapter in enumerate(model.adapters):
        spec, spec_mask = adapter(spec, spec_mask)
        print_shapes(
            f"Adapter[{i}] - {type(adapter).__name__}",
            dict(spec=spec, spec_mask=spec_mask),
        )

    for i, bg_adapter in enumerate(model.bg_adapters):
        bg_spec = bg_adapter(bg_spec)
        print_shapes(
            f"BgAdapter[{i}] - {type(bg_adapter).__name__}", dict(bg_spec=bg_spec)
        )

    if model.cfg.use_bg_spec:
        assert model.merger is not None
        bg_spec_mask = torch.ones_like(bg_spec).to(bg_spec.device)
        spec, spec_mask = model.merger(spec, spec_mask, bg_spec, bg_spec_mask)
        print_shapes("Merger", dict(spec=spec, spec_mask=spec_mask))

    if model.cfg.input_mask:
        spec = model.merge_spec_mask(spec, spec_mask)
    features = model.encoder(spec)
    print_shapes(
        "Encoder", {f"feature[{i}]": feature for i, feature in enumerate(features)}
    )
    x = model.decoder(features)
    print_shapes("Decoder", {"x": x})

    if model.recover_dual:
        x = model._do_recover_dual(x)
        print_shapes("recover dual", {"x": x})

    x = model.head(x)
    print_shapes("Head", {"x": x})


@torch.no_grad()
def get_2d_image(
    model: HmsModel,
    eeg: Tensor,
    cqf: Tensor,
    bg_spec: Tensor | None = None,
    device="cpu",
):
    """
    encoder手前の画像を確認する
    """
    model.train()
    model = model.to(device)

    output = model.feature_extractor(eeg, cqf)
    spec = output["spectrogram"]
    spec_mask = output["spec_mask"]
    if model.cfg.use_bg_spec and bg_spec is not None:
        bg_spec = rearrange(bg_spec, "b f t c -> b c f t")

    if model.spec_transform is not None:
        spec = model.spec_transform(spec)
        if model.cfg.use_bg_spec:
            bg_spec = model.spec_transform(bg_spec)

    for adapter in model.adapters:
        spec, spec_mask = adapter(spec, spec_mask)

    for bg_adapter in model.bg_adapters:
        bg_spec = bg_adapter(bg_spec)

    if model.cfg.use_bg_spec and bg_spec is not None:
        assert model.merger is not None
        bg_spec_mask = torch.ones_like(bg_spec).to(bg_spec.device)
        spec, spec_mask = model.merger(spec, spec_mask, bg_spec, bg_spec_mask)

    if model.cfg.input_mask:
        spec = model.merge_spec_mask(spec, spec_mask)

    return spec
