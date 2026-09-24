"""Seed, thiết bị, checkpoint và DataLoader dùng chung."""

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Bản này hỗ trợ CPU/CUDA; thiết bị khác chưa kiểm thử")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA chưa sẵn sàng; chọn device=cpu hoặc cài PyTorch CUDA")
    return device


def make_loader(dataset, cfg, training, epoch=0, balanced=False):
    generator = torch.Generator().manual_seed(cfg.seed + epoch)
    sampler = None
    if training and balanced:
        counts = np.bincount(dataset.labels)
        weights = [1.0 / counts[label] for label in dataset.labels]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=training and sampler is None,
                      sampler=sampler, num_workers=cfg.num_workers, generator=generator,
                      pin_memory=torch.cuda.is_available(), drop_last=False,
                      persistent_workers=False)


def lr_schedule(optimizer, total_steps, warmup_steps, minimum):
    warmup_steps = min(warmup_steps, max(0, total_steps - 1))

    def factor(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
        progress = min(1.0, max(0.0, progress))
        return minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def optimizer_update(model, optimizer, scaler, scheduler, sample_count, clip_grad):
    # Backward dùng tổng loss; chia gradient sau unscale => đúng cả batch cuối ngắn.
    scaler.unscale_(optimizer)
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(sample_count)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
    if not scaler.is_enabled() and not torch.isfinite(norm):
        raise FloatingPointError("Gradient NaN/Inf trên FP32")
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    if scaler.get_scale() >= old_scale:  # Không đẩy scheduler khi AMP bỏ qua update.
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    # Chỉ chấp nhận schema của project, không chạy pickle Python tùy ý.
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or value.get("format_version") != 1:
        raise ValueError("Checkpoint không đúng định dạng clean v1")
    return value


def rng_state():
    return {"torch": torch.get_rng_state(), "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(value):
    torch.set_rng_state(value["torch"])
    random.setstate(value["python"])
    if value["cuda"] and torch.cuda.is_available():
        if len(value["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume RNG cần cùng số GPU; dùng --pretrained để khởi tạo run mới")
        torch.cuda.set_rng_state_all(value["cuda"])


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
