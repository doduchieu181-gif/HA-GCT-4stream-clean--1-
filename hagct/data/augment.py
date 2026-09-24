"""Chỉ biến đổi frame thật; sau cùng pad cả dữ liệu và mask cùng vị trí."""

import numpy as np
from ..topology import PARENTS
from .preprocess import pad_sequence, resample_time


def scale_bones(x, factors):
    """Dùng vector xương GỐC; không tính lại từ parent vừa bị thay đổi."""
    bones = x - x[:, :, PARENTS]
    out = x.copy()
    # Thứ tự joint đảm bảo parent đã được xử lý trước child.
    for child in range(1, 27):
        out[:, :, child] = out[:, :, PARENTS[child]] + factors[child] * bones[:, :, child]
    return out


def augment_and_pad(x, cfg, max_frames, rng):
    x = x.copy()
    if cfg.enabled:
        if rng.random() < cfg.crop_probability:
            size = max(1, int(round(x.shape[1] * rng.uniform(cfg.crop_min_ratio, 1))))
            start = int(rng.integers(0, x.shape[1] - size + 1))
            x = x[:, start:start + size].copy()
        if rng.random() < cfg.speed_probability:
            # Tăng rate => ít frame hơn; KHÔNG resample rồi kéo ngược về T cũ.
            rate = rng.uniform(cfg.speed_min, cfg.speed_max)
            size = min(max_frames, max(1, round(x.shape[1] / rate)))
            x = resample_time(x, size)
        angle = np.deg2rad(rng.uniform(-cfg.rotation_degrees, cfg.rotation_degrees))
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32)
        x = np.einsum("cd,dtv->ctv", rotation, x)
        x = scale_bones(x, rng.uniform(1 - cfg.bone_scale, 1 + cfg.bone_scale, 27))
        x += rng.normal(0, cfg.noise_std, x.shape).astype(np.float32)
        # Mask sau noise để khớp đã che vẫn thật sự bằng 0.
        if rng.random() < cfg.mask_probability:
            joints = rng.choice(np.arange(1, 27), cfg.num_mask_joints, replace=False)
            x[:, :, joints] = 0
    offset = int(rng.integers(0, max_frames - x.shape[1] + 1)) if cfg.enabled and cfg.random_placement else 0
    return pad_sequence(x, max_frames, offset)
