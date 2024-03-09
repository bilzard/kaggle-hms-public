import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor


def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    """
    query: b n d
    key: b n d
    value: b n d
    ------------
    return: b n d
    """
    d_k = key.shape[-1] ** 0.5
    attn = torch.matmul(query, key.transpose(-1, -2)) / d_k
    attn = attn.softmax(dim=-1)
    return torch.matmul(attn, value)


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        merge_type: str = "concat",
        bottleneck_ratio: int = 1,
    ):
        super().__init__()
        input_dim = d_model
        d_model = d_model // bottleneck_ratio
        self.num_heads = num_heads
        self.qkv = nn.Linear(input_dim, d_model * 3)
        self.o = nn.Linear(
            d_model // num_heads if merge_type == "mean" else d_model, input_dim
        )
        self.merge_type = merge_type

    def forward(self, x: Tensor) -> Tensor:
        """
        x: b n d
        ------------
        return: b n d
        """
        qkv = self.qkv(x)
        qkv = rearrange(qkv, "b n (h d) -> b h n d", h=self.num_heads)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        x = scaled_dot_product_attention(q, k, v)  # b h n d
        if self.merge_type == "mean":
            x = x.mean(dim=1)
        else:
            x = rearrange(x, "b h n d -> b n (h d)")
        x = self.o(x)
        return x


class SharedQkMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        merge_type: str = "concat",
        bottleneck_ratio: int = 1,
    ):
        super().__init__()
        input_dim = d_model
        d_model = d_model // bottleneck_ratio

        self.num_heads = num_heads
        self.qv = nn.Linear(input_dim, d_model * 2)
        self.o = nn.Linear(
            d_model // num_heads if merge_type == "mean" else d_model, input_dim
        )
        self.merge_type = merge_type

    def forward(self, x: Tensor) -> Tensor:
        """
        x: b n d
        ------------
        return: b n d
        """
        qv = self.qv(x)
        qv = rearrange(qv, "b n (h d) -> b h n d", h=self.num_heads)
        q, v = torch.chunk(qv, 2, dim=-1)
        x = scaled_dot_product_attention(q, q, v)  # b h n d
        if self.merge_type == "mean":
            x = x.mean(dim=1)
        else:
            x = rearrange(x, "b h n d -> b n (h d)")
        x = self.o(x)
        return x


class SharedQkvMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        merge_type: str = "concat",
        bottleneck_ratio: int = 1,
    ):
        super().__init__()
        input_dim = d_model
        d_model = d_model // bottleneck_ratio
        self.num_heads = num_heads
        self.v = nn.Linear(input_dim, d_model)
        self.o = nn.Linear(
            d_model // num_heads if merge_type == "mean" else d_model, input_dim
        )
        self.merge_type = merge_type

    def forward(self, x: Tensor) -> Tensor:
        """
        x: b n d
        ------------
        return: b n d
        """
        v = self.v(x)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)
        x = scaled_dot_product_attention(v, v, v)  # b h n d
        if self.merge_type == "mean":
            x = x.mean(dim=1)
        else:
            x = rearrange(x, "b h n d -> b n (h d)")
        x = self.o(x)
        return x


if __name__ == "__main__":
    from torchinfo import summary

    batch_size = 2
    num_channels = 10
    d_model = 64
    num_heads = 4

    x = torch.rand(batch_size, num_channels, d_model)
    m = MultiHeadSelfAttention(d_model, num_heads=num_heads)
    y = m(x)
    print(y.shape)
    summary(m, input_size=(batch_size, num_channels, d_model))
    m = SharedQkMultiHeadSelfAttention(d_model, num_heads=num_heads)
    y = m(x)
    print(y.shape)
    summary(m, input_size=(batch_size, num_channels, d_model))
    m = SharedQkvMultiHeadSelfAttention(d_model, num_heads=num_heads)
    y = m(x)
    print(y.shape)
    summary(m, input_size=(batch_size, num_channels, d_model))
