"""Masked joint reconstruction: warm-start encoder, không dùng nhãn lớp."""

import torch
from torch import nn

from .encoder import HAGCTEncoder


class MaskedReconstruction(nn.Module):
    def __init__(self, cfg, max_frames):
        super().__init__()
        self.max_frames = max_frames
        self.encoder = HAGCTEncoder(cfg, max_frames)
        self.decoder = nn.Sequential(nn.Linear(cfg.d_model, 2 * cfg.d_model), nn.GELU(),
                                     nn.Linear(2 * cfg.d_model, 2 * max_frames))

    def forward(self, x, mask, masked_joints):
        features = self.encoder(x, mask, masked_joints)
        b, v, _ = features.shape
        prediction = self.decoder(features).reshape(b, v, self.max_frames, 2)
        return prediction.permute(0, 3, 2, 1)[:, :, :x.shape[2]]


def sample_joint_mask(batch, ratio, device, generator=None):
    if not 0 < ratio < 1:
        raise ValueError("mask_ratio thuộc (0,1)")
    count = min(26, max(1, round(27 * ratio)))
    order = torch.rand(batch, 27, device=device, generator=generator).argsort(dim=1)
    result = torch.zeros(batch, 27, dtype=torch.bool, device=device)
    return result.scatter_(1, order[:, :count], True)


def reconstruction_loss(prediction, target, frame_mask, joint_mask):
    """MSE mỗi mẫu, chỉ tính tọa độ của khớp bị che ở frame thật."""
    selected = frame_mask[:, None, :, None] & joint_mask[:, None, None, :]
    selected = selected.expand_as(target)
    error = (prediction - target).square().masked_fill(~selected, 0)
    return error.sum((1, 2, 3)) / selected.sum((1, 2, 3)).clamp_min(1)
