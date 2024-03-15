from torch import Tensor

from src.config import LossWeightConfig


def calc_weight_sum(weight: Tensor, cfg: LossWeightConfig) -> float:
    """
    weight: b k
    """
    assert weight.ndim == 2, f"Invalid shape: {weight.shape}"

    batch_size = weight.shape[0]
    match cfg.norm_policy:
        case "absolute":
            weight_sum = batch_size * cfg.global_mean
        case "relative":
            weight_sum = weight.sum(dim=0).mean().item()
        case _:
            raise ValueError(f"Invalid weight_norm_policy: {cfg.norm_policy}")

    return weight_sum
