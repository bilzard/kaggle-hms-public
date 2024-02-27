import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor


def collate_lr_channels(spec: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """
    左右のchannelをbatch方向に積み上げる
    Zチャネルは左右両方に入力する

    spec: (B, C, F, T)
    mask: (B, C, F, T)

    Return:
    spec: (2 * B, C, F, T)
    mask: (2 * B, C, F, T)
    """
    assert spec.shape[1] == 5

    spec_left = spec[:, [0, 1, 2], ...]
    spec_right = spec[:, [4, 3, 2], ...]
    spec = torch.cat([spec_left, spec_right], dim=0)

    mask_left = mask[:, [0, 1, 2], ...]
    mask_right = mask[:, [4, 3, 2], ...]
    mask = torch.cat([mask_left, mask_right], dim=0)

    return spec, mask


class WeightedMeanStackingAggregator(nn.Module):
    def __init__(self, norm_mask: bool = True, eps=1e-4):
        super().__init__()
        self.norm_mask = norm_mask
        self.eps = eps

    def forward(self, spec: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
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
            if self.norm_mask:
                sum_mask /= end - start
            sum_masks.append(sum_mask)

        specs = torch.concat(sum_specs, dim=1)
        masks = torch.concat(sum_masks, dim=1)

        return specs, masks


class DualWeightedMeanStackingAggregator(WeightedMeanStackingAggregator):
    def forward(self, spec: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        spec: (B, 20, H, W)
        mask: (B, 19, H, W)

        return:
        specs: (2B, 3, H, W)
        masks: (2B, 3, H, W)
        """
        spec, mask = super().forward(spec, mask)

        return collate_lr_channels(spec, mask)


class WeightedMeanTilingAggregator(WeightedMeanStackingAggregator):
    def __init__(self, norm_mask: bool = True, eps=1e-4):
        super().__init__(norm_mask=norm_mask, eps=eps)

    def forward(self, spec: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        spec: (B, 20, H, W)
        mask: (B, 19, H, W)

        return:
        specs: (2B, 1, 5H, W)
        masks: (2B, 1, 5H, W)
        """
        spec, mask = super().forward(spec, mask)
        spec = rearrange(spec, "b c h w -> b 1 (c h) w")
        mask = rearrange(mask, "b c h w -> b 1 (c h) w")
        return spec, mask


class DualWeightedMeanTilingAggregator(DualWeightedMeanStackingAggregator):
    def __init__(self, norm_mask: bool = True, eps=1e-4):
        super().__init__(norm_mask=norm_mask, eps=eps)

    def forward(self, spec: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        spec: (B, 20, H, W)
        mask: (B, 19, H, W)

        return:
        specs: (2B, 1, 3H, W)
        masks: (2B, 1, 3H, W)
        """
        spec, mask = super().forward(spec, mask)
        spec = rearrange(spec, "b c h w -> b 1 (c h) w")
        mask = rearrange(mask, "b c h w -> b 1 (c h) w")
        return spec, mask


class TilingAggregator(nn.Module):
    """
    周波数方向にspetrogramを積み上げる
    """

    def __init__(self):
        super().__init__()

    def forward(self, spec: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
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


class DualTilingAggregator(TilingAggregator):
    def forward(self, spec: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        spec, mask = super().forward(spec, mask)

        return collate_lr_channels(spec, mask)


def fill_canvas(spec: Tensor) -> Tensor:
    """
    全てのチャネルを1枚のキャンバスに敷き詰める

    input: (B 18 F T)
    output: (B 2 5F 2T)

    [
        [
            [LL0, LP0],
            [LL1, LP1],
            [LL2, LP2],
            [LL3, LP3],
            [Z0, Z1],
        ],
        [
            [RP0, RL0],
            [RP1, RL1],
            [RP2, RL2],
            [RP3, RL3],
            [Z0, Z1],
        ],
    ]
    """
    spec_ll = spec[:, 0:4]
    spec_lp = spec[:, 4:8]
    spec_z0 = spec[:, 8:9]
    spec_z1 = spec[:, 9:10]
    spec_rp = spec[:, 10:14]
    spec_rl = spec[:, 14:18]

    spec_l0 = torch.cat([spec_ll, spec_z0], dim=1)  # (B, 5, F, T)
    spec_l1 = torch.cat([spec_lp, spec_z1], dim=1)  # (B, 5, F, T)
    spec_l0 = rearrange(spec_l0, "b c f t -> b (c f) t")  # (B, 5F, T)
    spec_l1 = rearrange(spec_l1, "b c f t -> b (c f) t")  # (B, 5F, T)
    spec_l = torch.stack([spec_l0, spec_l1], dim=1)  # (B, 2, 5F, T)
    spec_l = rearrange(spec_l, "b c f t -> b f (c t)")  # (B, 5F, 2T)

    spec_r0 = torch.cat([spec_rl, spec_z0], dim=1)  # (B, 5, F, T)
    spec_r1 = torch.cat([spec_rp, spec_z1], dim=1)  # (B, 5, F, T)
    spec_r0 = rearrange(spec_r0, "b c f t -> b (c f) t")  # (B, 5F, T)
    spec_r1 = rearrange(spec_r1, "b c f t -> b (c f) t")  # (B, 5F, T)
    spec_r = torch.stack([spec_r0, spec_r1], dim=1)  # (B, 2, 5F, T)
    spec_r = rearrange(spec_r, "b c f t -> b f (c t)")  # (B, 5F, 2T)

    spec = torch.stack([spec_l, spec_r], dim=1)  # (B, 2, 5F, 2T)

    return spec


def fill_canvas_tr(spec: Tensor) -> Tensor:
    """
    全てのチャネルを1枚のキャンバスに敷き詰める

    input: (B 18 F T)
    output: (B 2 2F 5T)

    [
        [
            [LL0, LL1, LL2, LL3, Z0],
            [LP0, LP1, LP2, LP3, Z1],
        ],
        [
            [RP0, RP1, RP2, RP3, Z0],
            [RL0, RL1, RL2, RL3, Z1],
        ],
    ]
    """
    spec_ll = spec[:, 0:4]
    spec_lp = spec[:, 4:8]
    spec_z0 = spec[:, 8:9]
    spec_z1 = spec[:, 9:10]
    spec_rp = spec[:, 10:14]
    spec_rl = spec[:, 14:18]

    spec_l0 = torch.cat([spec_ll, spec_z0], dim=1)  # (B, 5, F, T)
    spec_l1 = torch.cat([spec_lp, spec_z1], dim=1)  # (B, 5, F, T)
    spec_l0 = rearrange(spec_l0, "b c f t -> b f (c t)")  # (B, F, 5T)
    spec_l1 = rearrange(spec_l1, "b c f t -> b f (c t)")  # (B, F, 5T)
    spec_l = torch.stack([spec_l0, spec_l1], dim=1)  # (B, 2, F, 5T)
    spec_l = rearrange(spec_l, "b c f t -> b (c f) t")  # (B, 2F, 5T)

    spec_r0 = torch.cat([spec_rl, spec_z0], dim=1)  # (B, 5, F, T)
    spec_r1 = torch.cat([spec_rp, spec_z1], dim=1)  # (B, 5, F, T)
    spec_r0 = rearrange(spec_r0, "b c f t -> b f (c t)")  # (B, F, 5T)
    spec_r1 = rearrange(spec_r1, "b c f t -> b f (c t)")  # (B, F, 5T)
    spec_r = torch.stack([spec_r0, spec_r1], dim=1)  # (B, 2, F, 5T)
    spec_r = rearrange(spec_r, "b c f t -> b (c f) t")  # (B, 2F, 5T)

    spec = torch.stack([spec_l, spec_r], dim=1)  # (B, 2, 2F, 5T)

    return spec


class CanvasAggregator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, spec: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        input: (B, 18, F, T)
        output: (B, 2, 5F, 2T)
        """
        spec = fill_canvas(spec)  # (B, 2, 5F, 2T)
        mask = fill_canvas(mask)  # (B, 2, 5F, 2T)

        return spec, mask


class DualCanvasAggregator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, spec: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        input: (B, 18, F, T)
        output: (2B, 1, 5F, 2T)
        """
        spec = fill_canvas(spec)  # (B, 2, 5F, 2T)
        mask = fill_canvas(mask)  # (B, 2, 5F, 2T)

        spec = rearrange(spec, "b c f t -> (c b) 1 f t")
        mask = rearrange(mask, "b c f t -> (c b) 1 f t")

        return spec, mask


class TransposedCanvasAggregator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, spec: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        input: (B, 18, F, T)
        output: (B, 2, 2F, 5T)
        """
        spec = fill_canvas_tr(spec)  # (B, 2, 2F, 5T)
        mask = fill_canvas_tr(mask)  # (B, 2, 2F, 5T)

        return spec, mask


class DualTransposedCanvasAggregator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, spec: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        input: (B, 18, F, T)
        output: (2B, 1, 2F, 5T)
        """
        spec = fill_canvas_tr(spec)  # (B, 2, 2F, 5T)
        mask = fill_canvas_tr(mask)  # (B, 2, 2F, 5T)

        spec = rearrange(spec, "b c f t -> (c b) 1 f t")
        mask = rearrange(mask, "b c f t -> (c b) 1 f t")

        return spec, mask


class FlatTilingAggregator(nn.Module):
    """
    周波数方向にspetrogramを積み上げる
    """

    def __init__(self):
        super().__init__()

    def forward(self, spec: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        B, C, F, T = spec.shape
        spec = rearrange(spec, "b c f t -> b 1 (c f) t")

        mask = mask.expand(B, C, F, T)
        mask = rearrange(mask, "b c f t -> b 1 (c f) t")

        return spec, mask
