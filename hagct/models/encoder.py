"""Một encoder = nhánh không gian HA-GC + nhánh thời gian MHSA."""

import torch
from torch import nn

from .blocks import (LocalTemporal, SpatialBlock, TemporalBlock,
                     apply_mask, masked_time_pool)


class HAGCTEncoder(nn.Module):
    def __init__(self, cfg, max_frames):
        super().__init__()
        d = cfg.d_model
        self.max_frames = max_frames
        self.input_projection = nn.Linear(2, d)
        self.joint_position = nn.Parameter(torch.randn(1, 1, 27, d) * 0.02)
        self.frame_position = nn.Parameter(torch.randn(1, max_frames, 1, d) * 0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, d))
        self.spatial = nn.ModuleList([
            SpatialBlock(d, cfg.dropout, cfg.drop_path * (i + 1) / cfg.spatial_layers)
            for i in range(cfg.spatial_layers)])
        self.spatial_time = LocalTemporal(d, cfg.dropout)
        self.spatial_pool = nn.Linear(2 * d, d)
        self.joint_score = nn.Sequential(nn.Linear(d, max(1, d // 4)), nn.Tanh(),
                                         nn.Linear(max(1, d // 4), 1))
        self.temporal_local = LocalTemporal(d, cfg.dropout)
        self.temporal = nn.ModuleList([
            TemporalBlock(d, cfg.num_heads, cfg.dropout,
                          cfg.drop_path * (i + 1) / cfg.temporal_layers)
            for i in range(cfg.temporal_layers)])
        self.temporal_pool = nn.Linear(2 * d, d)
        self.fusion_gate = nn.Linear(2 * d, d)
        self.output_norm = nn.LayerNorm(d)

    def forward(self, x, mask, masked_joints=None):
        """x=(B,2,T,27), mask=(B,T) -> (B,27,D).

        Pretrain: masked_joints=(B,27), mask token vẫn nhận positional embedding.
        """
        if x.ndim != 4 or x.shape[1] != 2 or x.shape[3] != 27:
            raise ValueError("Encoder cần x=(B,2,T,27)")
        b, _, t, v = x.shape
        if mask.shape != (b, t) or mask.dtype != torch.bool or t > self.max_frames:
            raise ValueError("Mask phải bool (B,T); T không vượt max_frames")
        if not mask.any(dim=1).all():
            raise ValueError("Mỗi mẫu phải có ít nhất một frame hợp lệ")
        raw = apply_mask(x.permute(0, 2, 3, 1), mask)
        z = self.input_projection(raw)
        if masked_joints is not None:
            if masked_joints.shape != (b, v) or masked_joints.dtype != torch.bool:
                raise ValueError("masked_joints phải bool (B,27)")
            z = torch.where(masked_joints[:, None, :, None], self.mask_token, z)
        z = apply_mask(z + self.joint_position + self.frame_position[:, :t], mask)

        # Nhánh S: giao tiếp giữa khớp; convolution thời gian cục bộ cho mỗi khớp.
        s = z
        for block in self.spatial:
            s = block(s, mask)
        local = s.permute(0, 2, 1, 3).reshape(b * v, t, -1)
        local_mask = mask[:, None].expand(-1, v, -1).reshape(b * v, t)
        local = self.spatial_time(local, local_mask)
        s = local.reshape(b, v, t, -1).permute(0, 2, 1, 3)
        s = self.spatial_pool(masked_time_pool(s, mask))  # B,V,D

        # Nhánh T nhận embedding ban đầu, pool khớp rồi attention giữa các frame.
        weights = self.joint_score(z).softmax(dim=2)
        temporal = apply_mask((weights * z).sum(dim=2), mask)
        temporal = self.temporal_local(temporal, mask)
        for block in self.temporal:
            temporal = block(temporal, mask)
        temporal = self.temporal_pool(masked_time_pool(temporal, mask))
        temporal = temporal[:, None].expand(-1, v, -1)

        # Gated fusion KHÔNG phải cross-attention hai chiều.
        gate = self.fusion_gate(torch.cat((s, temporal), dim=-1)).sigmoid()
        return self.output_norm(gate * s + (1 - gate) * temporal)


class ClassificationHead(nn.Module):
    def __init__(self, dim, classes, dropout):
        super().__init__()
        self.layers = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim),
                                    nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, classes))

    def forward(self, features):
        pooled = torch.cat((features.mean(dim=1), features.amax(dim=1)), dim=-1)
        return self.layers(pooled)
