"""One Euro Filter: cutoff thích ứng theo tốc độ, dùng dt thực tế.

Tham khảo công thức: https://gery.casiez.net/1euro/
Không lọc confidence như một tọa độ; mỗi clip tạo bộ lọc mới.
"""

import numpy as np


def smoothing_alpha(dt, cutoff):
    if dt <= 0 or np.any(np.asarray(cutoff) <= 0):
        raise ValueError("dt và cutoff phải dương")
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return dt / (dt + tau)


def one_euro_filter(coords, fps=30.0, min_cutoff=1.0, beta=0.7,
                    derivative_cutoff=1.0, timestamps=None):
    """coords: (T,V,2). Tính vector hóa theo joint và channel, lặp theo frame."""
    x = np.asarray(coords, dtype=np.float32)
    if x.ndim != 3 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("Bộ lọc cần (T,V,C) hữu hạn và T > 0")
    if fps <= 0 or min_cutoff <= 0 or derivative_cutoff <= 0 or beta < 0:
        raise ValueError("Tham số One Euro không hợp lệ")
    times = np.arange(len(x), dtype=np.float64) / fps if timestamps is None else np.asarray(timestamps)
    if times.shape != (len(x),) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("timestamps phải hữu hạn, tăng chặt và cùng độ dài T")
    result = np.empty_like(x)
    result[0] = x[0]
    derivative = np.zeros_like(x[0])
    for t in range(1, len(x)):
        dt = float(times[t] - times[t - 1])
        # Đạo hàm theo hai mẫu raw liên tiếp, không theo hai mẫu đã lọc.
        raw_derivative = (x[t] - x[t - 1]) / dt
        ad = smoothing_alpha(dt, derivative_cutoff)
        derivative = ad * raw_derivative + (1 - ad) * derivative
        cutoff = min_cutoff + beta * np.abs(derivative)
        alpha = smoothing_alpha(dt, cutoff)
        result[t] = alpha * x[t] + (1 - alpha) * result[t - 1]
    return result
