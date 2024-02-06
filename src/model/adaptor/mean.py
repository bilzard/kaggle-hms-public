import torch
import torch.nn as nn


class Mean(nn.Module):
    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def forward(
        self, spec: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
