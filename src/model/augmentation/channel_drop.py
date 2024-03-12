import torch
from torch import Tensor

from src.model.augmentation.base import BaseAugmentation


def channel_drop_spec(
    spec: Tensor,
    spec_mask: Tensor,
    drop_rate: float,
    batch_prob: float,
    spec_fill_value: float = -60.0,
    mask_fill_value: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """
    spec: b c t f
    spec_mask: b c t f
    """
    spec, spec_mask = spec.clone(), spec_mask.clone()

    batch_size, ch = spec.shape[:2]

    for b in range(batch_size):
        if torch.rand(1) < batch_prob:
            shuffled_idxs = torch.randperm(ch)
            spec[b, shuffled_idxs[: int(ch * drop_rate)]] = spec_fill_value
            spec_mask[b, shuffled_idxs[: int(ch * drop_rate)]] = mask_fill_value

    return spec, spec_mask


class ChannelDrop(BaseAugmentation):
    def __init__(
        self,
        batch_prob: float,
        drop_rate: float,
        p: float = 1,
    ):
        super().__init__(p=1.0)
        self.drop_rate = drop_rate
        self.batch_prob = batch_prob

    def apply(self, batch: dict[str, Tensor], output: dict[str, Tensor]) -> None:
        (
            output["spec"],
            output["spec_mask"],
        ) = channel_drop_spec(
            output["spec"],
            output["spec_mask"],
            drop_rate=self.drop_rate,
            batch_prob=self.batch_prob,
        )
