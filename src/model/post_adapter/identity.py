import torch.nn as nn
from torch import Tensor


class IdentityPostAdapter(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logit: Tensor) -> Tensor:
        return logit


if __name__ == "__main__":
    import torch

    model = IdentityPostAdapter()
    input = torch.randn(2, 3, 6)
    output = model(input)
    assert torch.allclose(input, output, atol=1e-6)
    print("ok!")
