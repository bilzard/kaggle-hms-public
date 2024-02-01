import torch
import torch.nn as nn


def same_padding_1d(
    x: torch.Tensor, kernel_size: int, stride: int, **kwargs
) -> torch.Tensor:
    pad_size = kernel_size - stride
    pad_left = pad_size // 2
    pad_right = pad_size - pad_left
    return nn.functional.pad(x, (pad_left, pad_right), **kwargs)


def rolling_mean(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    apply_padding: bool = True,
    no_grad: bool = True,
) -> torch.Tensor:
    """
    x: (B, C, T)
    """
    B, C, T = x.shape
    pad_size = (kernel_size - stride) // 2 if apply_padding else 0

    conv = nn.Conv1d(
        in_channels=C,
        out_channels=C,
        kernel_size=kernel_size,
        stride=stride,
        padding=pad_size,
        bias=False,
        groups=C,
    )
    conv = conv.to(x.device)
    conv.weight.data.fill_(1.0 / kernel_size)
    if no_grad:
        with torch.no_grad():
            result = conv(x)
    else:
        result = conv(x)
    return result
