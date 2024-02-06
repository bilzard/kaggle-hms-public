import timm
import torch.nn as nn


class TimmEncoder(nn.Module):
    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        in_channels: int = 3,
        depth: int = 5,
        **kwargs,
    ):
        super().__init__()
        kwargs = dict(
            in_chans=in_channels,
            features_only=True,
            pretrained=pretrained,
            out_indices=tuple(range(depth)),
            **kwargs,
        )

        self.model = timm.create_model(name, **kwargs)  # type: ignore

        self._in_channels = in_channels
        self._out_channels = [
            in_channels,
        ] + self.model.feature_info.channels()
        self._depth = depth

    def forward(self, x):
        features = self.model(x)
        features = [
            x,
        ] + features
        return features

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def output_stride(self):
        return min(self._output_stride, 2**self._depth)
