"""Evaluate và predict dùng lại đúng preprocessing lưu trong checkpoint."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from hagct.config import Config
from hagct.data.dataset import SkeletonDataset, read_manifest
from hagct.data.preprocess import pad_sequence, preprocess
from hagct.models import FourStreamHAGCT
from .common import choose_device, load_checkpoint, make_loader, write_json
from .metrics import classification_metrics


def load_model(path, device):
    checkpoint = load_checkpoint(path)
    if checkpoint["kind"] != "four_stream":
        raise ValueError("Evaluate/predict cần checkpoint four_stream, không phải encoder pretrain")
    cfg = Config.from_dict(checkpoint["config"])
    model = FourStreamHAGCT(cfg.model, cfg.data.max_frames).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, cfg


def class_names(path, count):
    if path is None:
        return [str(i) for i in range(count)]
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list) or len(value) != count or not all(isinstance(x, str) for x in value):
        raise ValueError("classes.json phải là list tên lớp theo thứ tự label, đúng num_classes")
    return value


def evaluate_main():
    parser = argparse.ArgumentParser(description="Đánh giá checkpoint trên val hoặc test")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", help="Đường dẫn thay thế nếu chuyển máy")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Thư mục đánh giá đã có dữ liệu: {output}")
    device = choose_device(args.device)
    model, cfg = load_model(args.checkpoint, device)
    manifest = args.manifest or cfg.data.manifest
    rows = read_manifest(manifest, cfg.model.num_classes, cfg.data.signer_disjoint)
    dataset = SkeletonDataset(rows, cfg.data, cfg.augment, args.split, training=False)
    loader = make_loader(dataset, cfg.train, False)
    labels, scores = [], []
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch["x"].to(device), batch["mask"].to(device))
            labels.append(batch["label"].numpy())
            scores.append(logits.float().cpu().numpy())
    labels, scores = np.concatenate(labels), np.concatenate(scores)
    metrics, confusion, per_class = classification_metrics(labels, scores, cfg.model.num_classes)
    metrics.update({"split": args.split, "checkpoint": str(Path(args.checkpoint).resolve()),
                    "fusion_weights": model.fusion_weights().detach().cpu().tolist()})
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "metrics.json", metrics)
    np.save(output / "confusion.npy", confusion, allow_pickle=False)
    np.save(output / "logits.npy", scores, allow_pickle=False)
    with (output / "per_class.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_class[0]))
        writer.writeheader()
        writer.writerows(per_class)
    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "true_label", "predicted_label"])
        writer.writerows((row["path"], int(label), int(score.argmax()))
                        for row, label, score in zip(dataset.rows, labels, scores))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def predict_main():
    parser = argparse.ArgumentParser(description="Dự đoán 1 file skeleton .npy")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--length", type=int, help="Số frame đầu hợp lệ nếu file đã right-pad")
    parser.add_argument("--classes", help="classes.json: list tên lớp đúng thứ tự label")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int)
    args = parser.parse_args()
    if args.top_k < 1:
        raise ValueError("top-k phải dương")
    if args.threads:
        torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    model, cfg = load_model(args.checkpoint, device)
    names = class_names(args.classes, cfg.model.num_classes)
    raw = np.load(args.input, allow_pickle=False)
    x = preprocess(raw, cfg.data, args.length)
    x, mask = pad_sequence(x, cfg.data.max_frames)
    with torch.inference_mode():
        logits, streams = model(torch.from_numpy(x)[None].to(device),
                                 torch.from_numpy(mask)[None].to(device), return_streams=True)
        probabilities, indices = logits.softmax(dim=-1).topk(min(args.top_k, cfg.model.num_classes))
    result = {"top_k": [{"label": int(index), "name": names[index], "probability": float(prob)}
                         for prob, index in zip(probabilities[0].cpu(), indices[0].cpu())],
              "stream_top1": {name: int(value.argmax(-1)) for name, value in streams.items()},
              "fusion_weights": model.fusion_weights().detach().cpu().tolist()}
    print(json.dumps(result, ensure_ascii=False, indent=2))
