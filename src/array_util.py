import numpy as np


def find_nearest_multiple(n, m):
    """
    find the smallest number that is multiple of m and greater than or equal to  n
    """
    assert n >= m >= 0, (n, m)
    return n + (m - n % m) % m


def pad_multiple_of(xs: np.ndarray, divisor: int, pad_value=0) -> np.ndarray:
    """Pad a array with zeros so that its length is a multiple of divisor.

    Args:
        xs: A 1-D array.
        divisor: The divisor of the length of the tensor.
        pad_value: The value to pad the tensor with.

    Returns:
        The padded array.
    """
    length = find_nearest_multiple(xs.shape[0], divisor)
    pad_size = length - xs.shape[0]
    return np.pad(xs, [[0, pad_size]], constant_values=pad_value)
