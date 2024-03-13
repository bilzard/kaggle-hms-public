from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .timm_encoder import TimmEncoder


def get_output_channels(model: nn.Module):
    """
    efficientnetの各blockの出力チャネル数を取得する
    """
    output_channels = []
    for block in [model.conv_stem] + list(model.blocks):
        for module in list(block.modules())[::-1]:
            if hasattr(module, "out_channels"):
                output_channels.append(module.out_channels)
                break
            elif hasattr(module, "num_features"):
                output_channels.append(module.num_features)
                break
            else:
                continue

    return output_channels


class EfficientNetEncoder(TimmEncoder):
    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        in_channels: int = 3,
        depth: int = 5,
        grad_checkpointing: bool = False,
        num_eeg_channels: int = 8,
        post_stem_adapter_factory: Callable | None = None,
        post_block_adapter_factory: Callable | None = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            pretrained=pretrained,
            in_channels=in_channels,
            depth=depth,
            grad_checkpointing=grad_checkpointing,
            **kwargs,
        )

        channels = get_output_channels(self.model)
        self.add_module(
            "post_stem_adapter",
            post_stem_adapter_factory(
                num_eeg_channels=num_eeg_channels,
                in_channels=channels[0],
                out_channels=channels[0],
            )
            if post_stem_adapter_factory
            else nn.Identity(),
        )
        self.add_module(
            "post_block_adapters",
            nn.ModuleList(
                post_block_adapter_factory(
                    num_eeg_channels=num_eeg_channels,
                    in_channels=c,
                    out_channels=c,
                )
                if post_block_adapter_factory
                else nn.Identity()
                for i, c in enumerate(channels[1:])
            ),
        )

    def forward_org(self, x: Tensor) -> list[Tensor]:
        x = self.model.conv_stem(x)
        x = self.model.bn1(x)
        x = self.post_stem_adapter(x)

        if self.model.feature_hooks is None:
            features = []
            if 0 in self.model._stage_out_idx:
                features.append(x)  # add stem out
            for i, (b, a) in enumerate(
                zip(self.model.blocks, self.post_block_adapters)
            ):
                if self.model.grad_checkpointing and not torch.jit.is_scripting():  # type: ignore
                    x = checkpoint(b, x)  # type: ignore
                else:
                    x = b(x)

                x = a(x)
                if i + 1 in self.model._stage_out_idx:
                    features.append(x)
            return features
        else:
            self.model.blocks(x)
            out = self.model.feature_hooks.get_output(x.device)
            return list(out.values())

    def forward(self, x):
        features = self.forward_org(x)
        features = [
            x,
        ] + features
        return features
