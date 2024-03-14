import re
from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn
from hydra.utils import call, instantiate

from src.config import ArchitectureConfig, LrAdjustmentConfig


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.current: float = 0.0
        self.mean: float = 0.0
        self.sum: float = 0.0
        self.count: float = 0

    def update(self, val: float, count: float):
        self.current = val
        self.sum += val * count
        self.count += count
        self.mean = self.sum / self.count


def move_device(x: dict[str, torch.Tensor], input_keys: list[str], device: str):
    for k, v in x.items():
        if k in input_keys:
            x[k] = v.to(device)


def get_model(cfg: ArchitectureConfig, **kwargs) -> nn.Module:
    model = instantiate(cfg.model_class, _partial_=True)(cfg, **kwargs)
    return model


def check_model(cfg: ArchitectureConfig, model: nn.Module, **kwargs) -> nn.Module:
    model_checker = call(cfg.model_checker, model=model, **kwargs)
    return model_checker


def adjust_lr(lr: float, batch_size: int) -> float:
    return lr * batch_size / 32.0


def get_lr_params(
    model: nn.Module,
    base_lr: float,
    lr_adjustments: list[LrAdjustmentConfig],
    verbose=True,
    no_decay_bias_params=True,
) -> list[dict[str, Any]]:
    param_groups: dict[str, Any] = defaultdict(
        lambda: {"params": [], "params_no_decay": [], "lr": base_lr}
    )

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # Skip frozen parameters

        assert isinstance(param.data, torch.Tensor)

        # Find the matching learning rate
        for item in lr_adjustments:
            pattern, ratio = re.compile(str(item.pattern)), float(item.ratio)

            if re.search(pattern, name):
                if "weight" in name or not no_decay_bias_params:
                    param_groups[pattern]["params"].append(param)  # type: ignore
                else:
                    param_groups[pattern]["params_no_decay"].append(param)  # type: ignore
                param_groups[pattern]["lr"] = base_lr * ratio  # type: ignore
                break
        else:  # Default group
            if "weight" in name or not no_decay_bias_params:
                param_groups["default"]["params"].append(param)  # type: ignore
            else:
                param_groups["default"]["params_no_decay"].append(param)  # type: ignore

    # Prepare final param_groups list for optimizer
    final_param_groups = []
    for group_info in param_groups.values():
        if group_info["params"]:
            final_param_groups.append(
                {"params": group_info["params"], "lr": group_info["lr"]}
            )
        if group_info["params_no_decay"]:
            final_param_groups.append(
                {
                    "params": group_info["params_no_decay"],
                    "lr": group_info["lr"],
                    "weight_decay": 0.0,
                }
            )

    if verbose:
        print("[INFO]: Parameter groups:")
        print("-" * 40)
        for i, (pattern, info) in enumerate(param_groups.items()):
            print(f"[{i}]: `{pattern}`")
            print("  -", "lr", info["lr"])
            print("  - #params_decay", len(info["params"]))  # type: ignore
            print("  - #params_no_decay", len(info["params_no_decay"]))  # type: ignore
            print("-" * 40)

    return final_param_groups
