import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class BinaryReductionPostAdapter(nn.Module):
    """
    多クラス分類モデルのlogitを、等価な2クラス分類モデルのlogitに変換する
    """

    def __init__(self, target_class_index: int = 0, num_classes: int = 6):
        super().__init__()
        self.target_class_index = target_class_index
        self.num_classes = num_classes

    def forward(self, logit: Tensor) -> Tensor:
        """
        logit: b k c
        """
        z0 = logit[..., self.target_class_index]

        n = logit.exp()
        n0 = n[..., self.target_class_index]

        z_bar = torch.log(n.sum(dim=-1) - n0) - np.log(5)

        logit = torch.zeros_like(logit).to(logit.device)
        logit[..., self.target_class_index] = z0 - z_bar

        return logit


if __name__ == "__main__":
    logit = torch.randn(2, 3, 6)
    post_adapter = BinaryReductionPostAdapter()
    logit_reduced = post_adapter(logit)
    print("logit: ", logit.detach().numpy().round(3))
    print("logit_reduced: ", logit_reduced.detach().numpy().round(3))
