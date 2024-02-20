import torch
import torch.nn as nn
from einops import rearrange


class WeightedMeanAggregator(nn.Module):
    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def forward(
        self, spec: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        spec: (B, 20, H, W)
        mask: (B, 19, H, W)

        return:
        specs: (B, 5, H, W)
        masks: (B, 5, H, W)
        """
        sum_specs = []
        sum_masks = []
        ranges = [0, 4, 8, 10, 14, 18]
        B, C, H, W = spec.shape
        for i, (start, end) in zip(range(C), zip(ranges[:-1], ranges[1:])):
            sum_spec = (spec[:, start:end] * mask[:, start:end]).sum(
                dim=1, keepdim=True
            )
            sum_mask = mask[:, start:end].sum(dim=1, keepdim=True)

            sum_specs.append(sum_spec / (sum_mask + self.eps))
            sum_masks.append(sum_mask)

        specs = torch.concat(sum_specs, dim=1)
        masks = torch.concat(sum_masks, dim=1)

        return specs, masks


class TilingAggregator(nn.Module):
    """
    周波数方向にspetrogramを積み上げる
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, spec: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tiled_specs = []
        tiled_masks = []
        ranges = [0, 4, 8, 10, 14, 18]
        B, C, F, T = spec.shape
        for i, (start, end) in zip(range(C), zip(ranges[:-1], ranges[1:])):
            ch = end - start
            pad_size = F * (4 - ch)
            tile = rearrange(spec[:, start:end], "b c f t -> b (c f) t", c=ch)
            tile = torch.nn.functional.pad(
                tile, (0, 0, 0, pad_size), mode="constant", value=0
            )
            tiled_specs.append(tile)

            mask = mask.expand(B, C, F, T)
            tiled_mask = rearrange(mask[:, start:end], "b c f t -> b (c f) t", c=ch)
            tiled_mask = torch.nn.functional.pad(
                tiled_mask, (0, 0, 0, pad_size), mode="constant", value=0
            )
            tiled_masks.append(tiled_mask)

        specs = torch.stack(tiled_specs, dim=1)
        masks = torch.stack(tiled_masks, dim=1)

        return specs, masks
