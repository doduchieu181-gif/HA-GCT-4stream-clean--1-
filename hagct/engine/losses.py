"""Loss trả vector (B,) để tích lũy gradient theo đúng số mẫu."""

import torch
from torch.nn import functional as F


def classification_loss(logits, labels, cfg):
    ce = F.cross_entropy(logits, labels, label_smoothing=cfg.label_smoothing, reduction="none")
    if cfg.loss == "ce":
        return ce
    # p_t lấy từ nhãn thật, không lấy exp(-CE đã label-smoothing).
    pt = logits.log_softmax(dim=-1).gather(1, labels[:, None]).squeeze(1).exp()
    return (1 - pt).pow(cfg.focal_gamma) * ce


def mixup_same_mask(x, mask, labels, alpha, rng):
    """Giữ ý nghĩa mask: chỉ ghép cặp có cùng các frame hợp lệ.

    Nếu một nhóm chỉ có 1 mẫu thì mẫu đó không thực sự được mix.
    """
    permutation = torch.arange(x.shape[0], device=x.device)
    if alpha <= 0:
        return x, labels, labels, 1.0
    _, groups = torch.unique(mask, dim=0, return_inverse=True)
    for group in groups.unique():
        indices = torch.where(groups == group)[0]
        permutation[indices] = indices[torch.randperm(len(indices), device=x.device)]
    weight = float(rng.beta(alpha, alpha))
    return weight * x + (1 - weight) * x[permutation], labels, labels[permutation], weight
