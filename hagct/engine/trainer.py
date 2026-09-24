"""Luồng train tường minh. Train/val dùng chung pipeline, test ở CLI riêng."""

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from hagct.config import load_config
from hagct.data.dataset import SkeletonDataset, manifest_hash, read_manifest
from hagct.models import FourStreamHAGCT
from hagct.models.pretrain import MaskedReconstruction, reconstruction_loss, sample_joint_mask
from .common import (atomic_save, choose_device, load_checkpoint, lr_schedule, make_loader,
                     optimizer_update, restore_rng, rng_state, seed_everything, write_json)
from .losses import classification_loss, mixup_same_mask
from .metrics import classification_metrics


def run_epoch(model, loader, cfg, device, training, optimizer=None, scaler=None,
              scheduler=None, pretrain=False, mask_ratio=0.3, epoch=0):
    model.train(training)
    losses, count, group_count = 0.0, 0, 0
    labels_all, scores_all = [], []
    if training:
        optimizer.zero_grad(set_to_none=True)
    # Local RNG tái lập theo epoch. Validation luôn cùng joint mask giữa các epoch.
    rng = np.random.default_rng(cfg.train.seed + epoch)
    mask_rng = torch.Generator(device=device).manual_seed(cfg.train.seed + (epoch if training else 0))
    amp_enabled = cfg.train.amp and device.type == "cuda"
    with torch.set_grad_enabled(training):
        for step, batch in enumerate(loader):
            x, mask, labels = (batch[key].to(device, non_blocking=True) for key in ("x", "mask", "label"))
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                if pretrain:
                    joint_mask = sample_joint_mask(len(x), mask_ratio, device, mask_rng)
                    predictions = model(x, mask, joint_mask)
                    loss_vector = reconstruction_loss(predictions.float(), x.float(), mask, joint_mask)
                else:
                    mixed, ya, yb, weight = mixup_same_mask(
                        x, mask, labels, cfg.train.mixup_alpha if training else 0, rng)
                    scores = model(mixed, mask)
                    loss_vector = weight * classification_loss(scores, ya, cfg.train)
                    loss_vector += (1 - weight) * classification_loss(scores, yb, cfg.train)
                    if not training:
                        labels_all.append(labels.cpu().numpy())
                        scores_all.append(scores.float().cpu().numpy())
            if not torch.isfinite(loss_vector).all():
                raise FloatingPointError(f"Loss NaN/Inf tại batch {step}")
            losses += loss_vector.detach().float().sum().item()
            count += len(x)
            if training:
                scaler.scale(loss_vector.sum()).backward()
                group_count += len(x)
                if (step + 1) % cfg.train.accum_steps == 0 or step + 1 == len(loader):
                    optimizer_update(model, optimizer, scaler, scheduler, group_count, cfg.train.clip_grad)
                    group_count = 0
    result = {"loss": losses / count, "samples": count}
    if not training and not pretrain:
        summary, _, _ = classification_metrics(np.concatenate(labels_all), np.concatenate(scores_all),
                                                 cfg.model.num_classes)
        result.update(summary)
    return result


def train(args, pretrain=False):
    cfg = load_config(args.config)
    if args.output:
        cfg.train.output_dir = args.output
    if args.device:
        cfg.train.device = args.device
    if args.epochs:
        cfg.train.epochs = args.epochs
    cfg.validate()
    if args.threads is not None:
        if args.threads < 1:
            raise ValueError("threads phải dương")
        torch.set_num_threads(args.threads)
    seed_everything(cfg.train.seed)
    device = choose_device(cfg.train.device)
    if args.resume and args.pretrained:
        raise ValueError("Chọn --resume HOẶC --pretrained, không dùng cùng lúc")
    if args.stop_after is not None and args.stop_after < 1:
        raise ValueError("stop-after phải >=1")
    if pretrain and not 0 < args.mask_ratio < 1:
        raise ValueError("mask-ratio thuộc (0,1)")
    rows = read_manifest(cfg.data.manifest, cfg.model.num_classes, cfg.data.signer_disjoint)
    # MSA tự che token ở đầu encoder. Không xóa joint trong reconstruction target.
    train_augment = replace(cfg.augment, mask_probability=0.0) if pretrain else cfg.augment
    train_set = SkeletonDataset(rows, cfg.data, train_augment, "train", cfg.train.seed)
    val_set = SkeletonDataset(rows, cfg.data, cfg.augment, "val", cfg.train.seed)
    missing = set(range(cfg.model.num_classes)) - set(train_set.labels)
    if missing and not pretrain:
        raise ValueError(f"Train thiếu lớp {sorted(missing)}; kiểm tra manifest/num_classes")
    # Không tạo test Dataset, không đọc tensor test khi chọn checkpoint.
    val_loader = make_loader(val_set, cfg.train, False)
    model = (MaskedReconstruction(cfg.model, cfg.data.max_frames) if pretrain else
             FourStreamHAGCT(cfg.model, cfg.data.max_frames)).to(device)
    if args.pretrained:
        if pretrain:
            raise ValueError("--pretrained chỉ dùng khi finetune; pretrain tiếp dùng --resume")
        initial = load_checkpoint(args.pretrained)
        if initial["kind"] != "encoder_pretrain":
            raise ValueError("--pretrained cần checkpoint encoder_pretrain")
        for key in ("d_model", "spatial_layers", "temporal_layers", "num_heads"):
            if initial["config"]["model"][key] != getattr(cfg.model, key):
                raise ValueError(f"Pretrained không cùng kiến trúc: {key}")
        encoder_state = {k.removeprefix("encoder."): v for k, v in initial["model"].items()
                         if k.startswith("encoder.")}
        print("Đã nạp encoder:", model.initialize_encoders(encoder_state), flush=True)

    head, backbone = [], []
    for name, parameter in model.named_parameters():
        (head if name.startswith("heads.") else backbone).append(parameter)
    groups = [{"params": backbone, "lr": cfg.train.lr}]
    if head:
        groups.append({"params": head, "lr": cfg.train.lr * cfg.train.classifier_lr_multiplier})
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg.train.weight_decay)
    updates_per_epoch = math.ceil(math.ceil(len(train_set) / cfg.train.batch_size) / cfg.train.accum_steps)
    scheduler = lr_schedule(optimizer, updates_per_epoch * cfg.train.epochs,
                            updates_per_epoch * cfg.train.warmup_epochs, cfg.train.min_lr_ratio)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.amp and device.type == "cuda")
    kind = "encoder_pretrain" if pretrain else "four_stream"
    fingerprint = manifest_hash(cfg.data.manifest)
    output = Path(cfg.train.output_dir).resolve()
    start_epoch, best, bad_epochs = 0, float("-inf"), 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        if checkpoint["kind"] != kind or checkpoint["manifest_sha256"] != fingerprint:
            raise ValueError("Resume khác loại model hoặc khác manifest")
        previous, current = checkpoint["config"], cfg.to_dict()
        # Chỉ cho đổi môi trường chạy; không âm thầm đổi bài toán/lịch LR khi resume.
        for key in ("device", "num_workers", "output_dir"):
            previous["train"].pop(key, None)
            current["train"].pop(key, None)
        if previous != current or checkpoint.get("mask_ratio") != (args.mask_ratio if pretrain else None):
            raise ValueError("Resume cần cùng cấu hình (kể cả epochs). Dùng checkpoint khởi tạo cho run mới.")
        if Path(args.resume).resolve().parent != output:
            raise ValueError("Resume phải dùng output_dir chứa checkpoint, để giữ log/best nhất quán")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng(checkpoint["rng"])
        start_epoch, best, bad_epochs = checkpoint["epoch"] + 1, checkpoint["best"], checkpoint["bad_epochs"]
        # Loại bản ghi sau epoch checkpoint nếu người dùng resume từ best cũ.
        log_path = output / "history.jsonl"
        if log_path.exists():
            entries = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
            if any(entry["epoch"] >= start_epoch for entry in entries):
                raise ValueError("Chỉ resume last.pt mới nhất; dùng run mới nếu muốn quay lại best.pt")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Không ghi đè run có sẵn: {output}; chọn --output khác hoặc --resume")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", cfg.to_dict())
    print(f"{kind} | {device} | params={sum(p.numel() for p in model.parameters()):,} | "
          f"train={len(train_set)}, val={len(val_set)} | output={output}", flush=True)
    if args.resume and bad_epochs >= cfg.train.patience:
        print("Run đã đạt early stopping; không tiếp tục optimizer. Tạo run mới cho thử nghiệm khác.")
        return output
    for epoch in range(start_epoch, cfg.train.epochs):
        train_set.set_epoch(epoch)
        loader = make_loader(train_set, cfg.train, True, epoch,
                             cfg.train.balanced_sampling and not pretrain)
        started = time.perf_counter()
        train_metrics = run_epoch(model, loader, cfg, device, True, optimizer, scaler, scheduler,
                                  pretrain, args.mask_ratio, epoch)
        val_metrics = run_epoch(model, val_loader, cfg, device, False, pretrain=pretrain,
                                mask_ratio=args.mask_ratio, epoch=epoch)
        score = -val_metrics["loss"] if pretrain else val_metrics["top1"]
        improved = score > best
        best, bad_epochs = (score, 0) if improved else (best, bad_epochs + 1)
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics,
                  "lr_next": [g["lr"] for g in optimizer.param_groups],
                  "seconds": time.perf_counter() - started}
        if not pretrain:
            record["fusion_weights"] = model.fusion_weights().detach().cpu().tolist()
        state = {"format_version": 1, "kind": kind, "epoch": epoch, "best": best,
                 "bad_epochs": bad_epochs, "config": cfg.to_dict(), "manifest_sha256": fingerprint,
                 "mask_ratio": args.mask_ratio if pretrain else None, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "scaler": scaler.state_dict(), "rng": rng_state()}
        if improved:
            atomic_save(state, output / "best.pt")
        atomic_save(state, output / "last.pt")
        with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if bad_epochs >= cfg.train.patience:
            print("Early stopping theo validation.", flush=True)
            break
        if args.stop_after is not None and epoch + 1 >= args.stop_after:
            print("Dừng có checkpoint theo --stop-after; có thể --resume.", flush=True)
            break
    return output


def main(pretrain=False):
    parser = argparse.ArgumentParser(description="Pretrain encoder" if pretrain else "Train HA-GCT 4 stream")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int, help="Override cho run mới, không đổi tổng epochs khi resume")
    parser.add_argument("--threads", type=int, help="CPU threads; demo nên đặt 1")
    parser.add_argument("--resume")
    parser.add_argument("--pretrained")
    parser.add_argument("--stop-after", type=int, help="Dừng sau N epoch tổng cộng nhưng giữ nguyên lịch LR")
    parser.add_argument("--mask-ratio", type=float, default=0.3)
    train(parser.parse_args(), pretrain=pretrain)
