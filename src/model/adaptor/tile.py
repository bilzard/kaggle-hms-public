import torch
import torch.nn as nn
from einops import rearrange


class Tile(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, spec: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tiled_specs = []
        tiled_masks = []
        ranges = [0, 4, 8, 10, 14, 18]
        B, C, H, W = spec.shape
        for i, (start, end) in zip(range(C), zip(ranges[:-1], ranges[1:])):
            ch = end - start
            pad_size = W * (4 - ch)
            tile = rearrange(spec[:, start:end], "b c h w -> b h (w c)", c=ch)
            tile = torch.nn.functional.pad(
                tile, (0, pad_size), mode="constant", value=0
            )
            tiled_specs.append(tile)

            tiled_mask = rearrange(mask[:, start:end], "b c h w -> b h (w c)", c=ch)
            tiled_mask = torch.nn.functional.pad(
                tiled_mask, (0, pad_size), mode="constant", value=0
            )
            tiled_masks.append(tiled_mask)

        specs = torch.stack(tiled_specs, dim=1)
        masks = torch.stack(tiled_masks, dim=1)

        return specs, masks
