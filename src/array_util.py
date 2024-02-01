import numpy as np


def find_nearest_multiple(n, m):
    """
    find the smallest number that is multiple of m and greater than or equal to  n
    """
    assert n >= m >= 0, (n, m)
    return n + (m - n % m) % m


def pad_multiple_of(
    xs: np.ndarray,
    divisor: int,
    axis: int = 0,
    pad_value: float = 0,
    padding_type="right",
) -> np.ndarray:
    """Pad a array with zeros so that its length is a multiple of divisor.
    padding is applied to the first dimension of the array.

    Args:
        xs: numpy array of N-dimensions
        divisor: The divisor of the length of the tensor
        axis: The axis to apply padding (default=0)
        pad_value: The value to pad the tensor with (default=0)

    Returns:
        The padded array.
    """
    length = find_nearest_multiple(xs.shape[axis], divisor)
    pad_size = length - xs.shape[axis]

    if padding_type == "right":
        left_pad = 0
        right_pad = pad_size
    elif padding_type == "left":
        left_pad = pad_size
        right_pad = 0
    elif padding_type == "both":
        left_pad = pad_size // 2
        right_pad = pad_size - left_pad
    else:
        raise ValueError(f"Unknown padding type: {padding_type}")

    return np.pad(
        xs,
        [(left_pad, right_pad) if i == axis else (0, 0) for i in range(xs.ndim)],
        constant_values=pad_value,
    )
