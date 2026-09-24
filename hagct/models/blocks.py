"""Các khối nhỏ. Quy ước feature: (B, T, V, D), mask: (B, T)."""

import torch
from torch import nn
from torch.nn import functional as F

from hagct.topology import graph_subsets


def apply_mask(x, mask):
    # masked_fill cũng loại được NaN ở padding; phép nhân với 0 thì không.
    shape = (*mask.shape, *((1,) * (x.ndim - 2)))
    return x.masked_fill(~mask.reshape(shape), 0)


def masked_time_pool(x, mask):
    """Ghép mean + max theo T, loại hoàn toàn frame đệm."""
    shape = (*mask.shape, *((1,) * (x.ndim - 2)))
    valid = mask.reshape(shape)
    mean = x.masked_fill(~valid, 0).sum(1) / valid.sum(1).clamp_min(1)
    maximum = x.masked_fill(~valid, -torch.inf).amax(1)
    return torch.cat((mean, maximum), dim=-1)


class DropPath(nn.Module):
    def __init__(self, probability=0.0):
        super().__init__()
        self.probability = probability

    def forward(self, x):
        if not self.training or self.probability == 0:
            return x
        keep = 1 - self.probability
        gate = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * gate / keep


class AdaptiveGraphConv(nn.Module):
    """Học trọng số trên cạnh có sẵn, không tự thêm cạnh ngoài topology.

    A[k, u, v] truyền từ khớp nguồn u đến khớp đích v.
    Ba subset: self, inward, outward. Chuẩn hóa tổng nguồn cho từng đích.
    """
    def __init__(self, dim, adjacency):
        super().__init__()
        self.register_buffer("support", torch.as_tensor(adjacency, dtype=torch.float32))
        self.edge_logits = nn.Parameter(torch.zeros_like(self.support))
        self.projection = nn.Linear(dim, 3 * dim, bias=False)
        self.dim = dim

    def adjacency(self):
        weights = F.softplus(self.edge_logits) * self.support
        return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def forward(self, x):
        b, t, v, _ = x.shape
        x = self.projection(x).reshape(b, t, v, 3, self.dim)
        return torch.einsum("btukd,kuv->btvd", x, self.adjacency()) / 3


class SpatialBlock(nn.Module):
    def __init__(self, dim, dropout, drop_path):
        super().__init__()
        body, hands = graph_subsets()
        self.norm = nn.LayerNorm(dim)
        self.body = AdaptiveGraphConv(dim, body)
        self.hands = AdaptiveGraphConv(dim, hands)
        self.hand_gate = nn.Parameter(torch.tensor(0.0))
        self.output = nn.Sequential(nn.GELU(), nn.Linear(dim, dim), nn.Dropout(dropout))
        self.drop_path = DropPath(drop_path)

    def forward(self, x, mask):
        z = self.norm(x)
        gate = self.hand_gate.sigmoid()
        z = (1 - gate) * self.body(z) + gate * self.hands(z)
        return apply_mask(x + self.drop_path(self.output(z)), mask)


class LocalTemporal(nn.Module):
    """Depthwise Conv theo T; LayerNorm không trộn thống kê với padding."""
    def __init__(self, dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.output = nn.Sequential(nn.GELU(), nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask):
        z = apply_mask(self.norm(x), mask)
        z = self.conv(z.transpose(1, 2)).transpose(1, 2)
        return apply_mask(x + self.output(z), mask)


class TemporalBlock(nn.Module):
    def __init__(self, dim, heads, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        # FFN chỉ trả transform; residual được cộng ĐÚNG MỘT LẦN ở forward.
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(4 * dim, dim), nn.Dropout(dropout))
        self.drop_path = DropPath(drop_path)

    def forward(self, x, mask):
        z = self.norm1(x)
        z, _ = self.attention(z, z, z, key_padding_mask=~mask, need_weights=False)
        x = apply_mask(x + self.drop_path(z), mask)
        return apply_mask(x + self.drop_path(self.ffn(self.norm2(x))), mask)
