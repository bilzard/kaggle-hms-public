import numpy as np


def find_nearest_multiple(n, m):
    """
    find the smallest number that is multiple of m and greater than or equal to  n
    """
    assert n >= m >= 0, (n, m)
    return n + (m - n % m) % m


def pad_multiple_of(
    xs: np.ndarray, divisor: int, axis: int = 0, pad_value=0
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
    return np.pad(
        xs,
        [(0, pad_size) if i == axis else (0, 0) for i in range(xs.ndim)],
        constant_values=pad_value,
    )
