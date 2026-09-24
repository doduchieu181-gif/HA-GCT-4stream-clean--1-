"""Tiền xử lý xác định: KHÔNG augmentation, KHÔNG học thống kê từ test.

raw → loại padding theo length → nội suy → One Euro → chuẩn hóa → CTV.
Padding thực hiện SAU augmentation để noise/crop không chạm vào padding.
"""

import numpy as np
from .filter import one_euro_filter


def resample_time(x, length):
    """Nội suy toàn bộ clip (C,T,V), không chỉ lấy các frame đầu."""
    if length < 1 or x.shape[1] < 1:
        raise ValueError("Chuỗi phải có ít nhất một frame")
    if length == x.shape[1]:
        return x.copy()
    old = np.arange(x.shape[1])
    new = np.linspace(0, x.shape[1] - 1, length)
    out = np.empty((x.shape[0], length, x.shape[2]), dtype=np.float32)
    for c in range(x.shape[0]):
        for v in range(x.shape[2]):
            out[c, :, v] = np.interp(new, old, x[c, :, v])
    return out


def preprocess(raw, cfg, valid_length=None):
    """Trả (2,T,27) chưa padding. Cấu hình channel phải được khai báo rõ."""
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 3:
        raise ValueError(f"Mỗi mẫu phải có 3 chiều, nhận {raw.shape}")
    x = raw.transpose(1, 2, 0) if cfg.layout == "CTV" else raw.copy()
    expected_c = 2 if cfg.channels == "xy" else 3
    if x.shape[1:] != (27, expected_c):
        raise ValueError(f"Cần TVC=(T,27,{expected_c}), nhận {x.shape}; kiểm tra layout/channels")
    length = x.shape[0] if valid_length is None else int(valid_length)
    if not 1 <= length <= x.shape[0]:
        raise ValueError(f"length={length} không hợp lệ với T={x.shape[0]}")
    x = x[:length]  # Chỉ bỏ padding được người dùng xác nhận bằng length
    coords = x[..., :2].copy()
    valid = np.isfinite(coords).all(axis=-1)
    if cfg.channels == "xy_conf":
        confidence = x[..., 2]
        finite_conf = confidence[np.isfinite(confidence)]
        if np.any((finite_conf < 0) | (finite_conf > 1)):
            raise ValueError("Confidence phải thuộc [0,1]; dữ liệu xyz cần channels=xyz")
        valid &= np.isfinite(confidence) & (confidence >= cfg.confidence_threshold)
    if cfg.zero_is_missing:
        valid &= np.any(coords != 0, axis=-1)
    observed = valid.any(axis=0)
    if not observed[cfg.root_joint]:
        raise ValueError("Root joint mất trong toàn bộ clip; cần sửa dữ liệu hoặc đổi root_joint")
    # Linear interpolation theo thời gian; đầu/cuối dùng giá trị gần nhất.
    # Joint mất toàn clip không thể suy ra đáng tin cậy: giữ 0 sau normalization.
    indices = np.arange(length)
    for v in range(27):
        keep = valid[:, v]
        for c in range(2):
            coords[:, v, c] = np.interp(indices, indices[keep], coords[keep, v, c]) if keep.any() else 0
    if cfg.filter_enabled:
        coords = one_euro_filter(coords, cfg.fps, cfg.min_cutoff, cfg.beta, cfg.derivative_cutoff)
    coords -= coords[:, cfg.root_joint:cfg.root_joint + 1, :].copy()
    if cfg.scale_normalization:
        if not observed[[1, 2]].all():
            raise ValueError("Không đủ hai vai để scale normalization")
        distances = np.linalg.norm(coords[:, 1] - coords[:, 2], axis=-1)
        good = distances[distances > 1e-6]
        if not len(good):
            raise ValueError("Khoảng cách hai vai bằng 0; kiểm tra thứ tự 27 joint")
        coords /= float(np.median(good))  # Một scale/clip, hạn chế rung do scale mỗi frame
    coords[:, ~observed] = 0
    result = coords.transpose(2, 0, 1).astype(np.float32)
    if result.shape[1] > cfg.max_frames:
        result = resample_time(result, cfg.max_frames)
    if not np.isfinite(result).all():
        raise ValueError("Tiền xử lý tạo NaN/Inf")
    return result


def pad_sequence(x, max_frames, offset=0):
    """Pad tọa độ VÀ tạo mask tương ứng ngay tại cùng một hàm."""
    t = x.shape[1]
    if t < 1 or t > max_frames or not 0 <= offset <= max_frames - t:
        raise ValueError("length/offset không hợp lệ khi padding")
    out = np.zeros((2, max_frames, 27), dtype=np.float32)
    mask = np.zeros(max_frames, dtype=bool)
    out[:, offset:offset + t] = x
    mask[offset:offset + t] = True
    return out, mask
