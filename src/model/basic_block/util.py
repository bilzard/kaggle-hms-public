import torch.nn.functional as F
from torch import Tensor


def calc_similarity(x: Tensor, y: Tensor, channel_dim: int = 1, eps=1e-4) -> Tensor:
    """
    x: (B, C, F, T)
    y: (B, C, F, T)

    Returns:
    similarity: (B, 1, F, T)
    """

    return F.cosine_similarity(x, y, dim=channel_dim, eps=eps).unsqueeze(channel_dim)
