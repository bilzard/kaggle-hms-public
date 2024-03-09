import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor


def collate_lr_channels(
    spec: Tensor, mask: Tensor, drop_z: bool = False
) -> tuple[Tensor, Tensor]:
    """
    左右のchannelをbatch方向に積み上げる
    Zチャネルは左右両方に入力する

    spec: (B, C, F, T)
    mask: (B, C, F, T)

    Return:
    spec: (2B, C, F, T)
    mask: (2B, C, F, T)
    """
    left_idxs = [0, 1]
    right_idxs = [4, 3]
    if not drop_z:
        left_idxs.append(2)
        right_idxs.append(2)

    spec_left = spec[:, left_idxs, ...]
    spec_right = spec[:, right_idxs, ...]
    spec = torch.cat([spec_left, spec_right], dim=0)

    mask_left = mask[:, left_idxs, ...]
    mask_right = mask[:, right_idxs, ...]
    mask = torch.cat([mask_left, mask_right], dim=0)

    return spec, mask


class WeightedMeanStackingAggregator(nn.Module):
    def __init__(self, drop_z: bool = False, norm_mask: bool = True, eps=1e-4):
        super().__init__()
        self.norm_mask = norm_mask
        self.eps = eps
        self.drop_z = drop_z

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
        for i, (start, end) in enumerate(zip(ranges[:-1], ranges[1:])):
            if self.drop_z and i == 2:
                continue
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
    def __init__(self, drop_z: bool = False, norm_mask: bool = True, eps=1e-4):
        super().__init__(drop_z=drop_z, norm_mask=norm_mask, eps=eps)

    def forward(self, spec: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        spec: (B, 20, H, W)
        mask: (B, 19, H, W)

        return:
        specs: (2B, 3, H, W)
        masks: (2B, 3, H, W)
        """
        spec, mask = super().forward(spec, mask)

        return collate_lr_channels(spec, mask, drop_z=self.drop_z)


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
    def __init__(self, drop_z: bool = False, norm_mask: bool = True, eps=1e-4):
        super().__init__(drop_z=drop_z, norm_mask=norm_mask, eps=eps)

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

    def __init__(self, drop_z: bool = False):
        super().__init__()
        self.drop_z = drop_z

    def forward(self, spec: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        tiled_specs = []
        tiled_masks = []
        ranges = [0, 4, 8, 10, 14, 18]
        B, C, F, T = spec.shape
        mask = mask.expand(B, C, F, T)

        for i, (start, end) in enumerate(zip(ranges[:-1], ranges[1:])):
            if i == 2 and self.drop_z:
                continue
            ch = end - start
            pad_size = F * (4 - ch)
            tile = rearrange(spec[:, start:end], "b c f t -> b (c f) t", c=ch)
            tile = torch.nn.functional.pad(
                tile, (0, 0, 0, pad_size), mode="constant", value=0
            )
            tiled_specs.append(tile)

            tiled_mask = rearrange(mask[:, start:end], "b c f t -> b (c f) t", c=ch)
            tiled_mask = torch.nn.functional.pad(
                tiled_mask, (0, 0, 0, pad_size), mode="constant", value=0
            )
            tiled_masks.append(tiled_mask)

        specs = torch.stack(tiled_specs, dim=1)
        masks = torch.stack(tiled_masks, dim=1)

        return specs, masks


class DualTilingAggregator(TilingAggregator):
    def __init__(self, drop_z: bool = False):
        super().__init__()
        self.drop_z = drop_z

    def forward(self, spec: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        spec, mask = super().forward(spec, mask)

        return collate_lr_channels(spec, mask, drop_z=self.drop_z)


def fill_canvas(spec: Tensor, drop_z: bool = False) -> Tensor:
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

    spec_l0, spec_l1 = spec_ll, spec_lp
    if not drop_z:
        spec_l0 = torch.cat([spec_l0, spec_z0], dim=1)  # (B, 5, F, T)
        spec_l1 = torch.cat([spec_l1, spec_z1], dim=1)  # (B, 5, F, T)

    spec_l0 = rearrange(spec_l0, "b c f t -> b (c f) t")  # (B, 5F, T)
    spec_l1 = rearrange(spec_l1, "b c f t -> b (c f) t")  # (B, 5F, T)

    spec_l = torch.stack([spec_l0, spec_l1], dim=1)  # (B, 2, 5F, T)
    spec_l = rearrange(spec_l, "b c f t -> b f (c t)")  # (B, 5F, 2T)

    spec_r0, spec_r1 = spec_rl, spec_rp
    if not drop_z:
        spec_r0 = torch.cat([spec_r0, spec_z0], dim=1)  # (B, 5, F, T)
        spec_r1 = torch.cat([spec_r1, spec_z1], dim=1)  # (B, 5, F, T)

    spec_r0 = rearrange(spec_r0, "b c f t -> b (c f) t")  # (B, 5F, T)
    spec_r1 = rearrange(spec_r1, "b c f t -> b (c f) t")  # (B, 5F, T)
    spec_r = torch.stack([spec_r0, spec_r1], dim=1)  # (B, 2, 5F, T)
    spec_r = rearrange(spec_r, "b c f t -> b f (c t)")  # (B, 5F, 2T)

    spec = torch.stack([spec_l, spec_r], dim=1)  # (B, 2, 5F, 2T)

    return spec


class CanvasAggregator(nn.Module):
    def __init__(self, drop_z: bool = False):
        super().__init__()
        self.drop_z = drop_z

    def forward(self, spec: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        input: (B, 18, F, T)
        output: (B, 2, 5F, 2T)
        """
        spec = fill_canvas(spec, drop_z=self.drop_z)  # (B, 2, 5F, 2T)
        mask = fill_canvas(mask, drop_z=self.drop_z)  # (B, 2, 5F, 2T)

        return spec, mask


class DualCanvasAggregator(CanvasAggregator):
    def __init__(self, drop_z: bool = False):
        super().__init__(drop_z=drop_z)

    def forward(self, spec: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        input: (B, 18, F, T)
        output: (2B, 1, 5F, 2T)
        """
        spec, mask = super().forward(spec, mask)

        spec = rearrange(spec, "b c f t -> (c b) 1 f t")
        mask = rearrange(mask, "b c f t -> (c b) 1 f t")

        return spec, mask


class CanvasWeightedMeanAggregator(CanvasAggregator):
    def __init__(self, drop_z: bool = False, norm_mask=True, eps=1e-4):
        super().__init__(drop_z=drop_z)
        self.eps = eps
        self.norm_mask = norm_mask

    def forward(self, spec: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        input: (B, 18, F, T)
        output: (B, 2, 5F, T)
        """
        spec, mask = super().forward(spec, mask)  # (B, 2, 5F, 2T)

        spec = rearrange(spec, "b c f (s t) -> b c f s t", s=2)  # (B, C, 5F, 2, T)
        mask = rearrange(mask, "b c f (s t) -> b c f s t", s=2)  # (B, C, 5F, 2, T)

        spec = (spec * mask).sum(dim=3)  # (B, C, 5F, T)
        mask = mask.sum(dim=3)  # (B, C, 5F, T)

        spec /= mask + self.eps
        if self.norm_mask:
            mask /= 2

        return spec, mask


class DualCanvasWeightedMeanAggregator(CanvasWeightedMeanAggregator):
    def __init__(self, drop_z: bool = False, norm_mask=True, eps=1e-4):
        super().__init__(drop_z=drop_z, norm_mask=norm_mask, eps=eps)

    def forward(self, spec: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        input: (B, 18, F, T)
        output: (2B, 1, 5F, T)
        """
        spec, mask = super().forward(spec, mask)  # (B, 2, 5F, T)
        spec = rearrange(spec, "b c f t -> (c b) 1 f t")
        mask = rearrange(mask, "b c f t -> (c b) 1 f t")

        return spec, mask


def collate_dual_separated_channels(x: Tensor, drop_z: bool = False) -> Tensor:
    """
    input: b ch f t
    output: (d ch b) 1 f t
    """
    x_ll = x[:, 0:4]
    x_lp = x[:, 4:8]
    x_z = x[:, 8:10]
    x_rp = x[:, 10:14]
    x_rl = x[:, 14:18]

    x_left = torch.cat([x_ll, x_lp], dim=1)
    x_right = torch.cat([x_rl, x_rp], dim=1)

    if not drop_z:
        x_left = torch.cat([x_left, x_z], dim=1)
        x_right = torch.cat([x_right, x_z], dim=1)

    x = torch.stack([x_left, x_right], dim=0)  # d b ch f t
    x = rearrange(x, "d b ch f t -> (d ch b) 1 f t").contiguous()

    return x


class DualChannelSeparatedAggregator(nn.Module):
    def __init__(self, drop_z: bool = False):
        super().__init__()
        self.drop_z = drop_z

    @torch.no_grad()
    def forward(self, spec: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        input: b ch f t
        output: (d ch b) 1 f t
        """
        B, C, F, T = spec.shape
        mask = mask.expand(B, C, F, T)

        spec = collate_dual_separated_channels(spec, drop_z=self.drop_z)
        mask = collate_dual_separated_channels(mask, drop_z=self.drop_z)

        return spec, mask
