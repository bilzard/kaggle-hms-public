import torch
import torch.nn as nn
from hydra.utils import call, instantiate

from src.config import ArchitectureConfig


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
