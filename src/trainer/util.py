from torch import Tensor

from src.config import LossWeightConfig


def calc_weight_sum(weight: Tensor, cfg: LossWeightConfig) -> float:
    batch_size = weight.shape[0]
    match cfg.norm_policy:
        case "absolute":
            weight_sum = batch_size * cfg.global_mean
        case "relative":
            weight_sum = weight.sum().item()
        case _:
            raise ValueError(f"Invalid weight_norm_policy: {cfg.norm_policy}")

    return weight_sum
