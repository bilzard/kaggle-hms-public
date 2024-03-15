import torch
import torch.nn as nn
import torch.nn.functional as F


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
) -> torch.Tensor:
    """
    x: (B, C, T)
    """
    _, C, _ = x.shape
    pad_size = (kernel_size - stride) // 2 if apply_padding else 0

    weight = torch.full(
        (C, 1, kernel_size),
        fill_value=1.0 / kernel_size,
        device=x.device,
        dtype=x.dtype,
    )
    result = F.conv1d(x, weight, stride=stride, padding=pad_size, groups=C)
    return result


if __name__ == "__main__":
    x = torch.arange(10).reshape(1, 1, -1).float()
    print("x:", x.detach().numpy().flatten())
    print("same_padding_1d", same_padding_1d(x, 3, 1).detach().numpy())
    print(
        "rolling_mean(k=3, s=1)",
        rolling_mean(
            same_padding_1d(x, 3, 1, mode="replicate"), 3, 1, apply_padding=False
        )
        .detach()
        .numpy(),
    )
