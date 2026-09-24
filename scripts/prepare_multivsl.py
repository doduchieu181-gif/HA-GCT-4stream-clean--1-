"""Tạo manifest từ source/<word_id>/<signer>_<clip>.npy, không di chuyển dữ liệu."""

import argparse
import csv
import os
from pathlib import Path

import numpy as np

from hagct.engine.common import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", choices=["signer", "random"], default="signer")
    parser.add_argument("--train-signers", nargs="+", default=[])
    parser.add_argument("--val-signers", nargs="+", default=[])
    parser.add_argument("--test-signers", nargs="+", default=[])
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    classes = sorted(p.name for p in source.iterdir() if p.is_dir() and any(p.glob("*.npy")))
    if not classes:
        raise ValueError("Không tìm thấy source/<word_id>/*.npy")
    mapping = {}
    if args.protocol == "signer":
        for split in ("train", "val", "test"):
            values = getattr(args, split + "_signers")
            if not values:
                raise ValueError(f"Cần --{split}-signers theo protocol bạn đã chọn")
            for signer in values:
                if signer in mapping:
                    raise ValueError(f"Signer trùng: {signer}")
                mapping[signer] = split
    elif not (0 < args.val_ratio < 1 and 0 < args.test_ratio < 1 and args.val_ratio + args.test_ratio < 1):
        raise ValueError("Ratios phải dương và tổng <1")
    else:
        print("CẢNH BÁO: random split KHÔNG phải cross-signer; không so trực tiếp hai protocol.")
    rng, rows, observed = np.random.default_rng(args.seed), [], set()
    for label, name in enumerate(classes):
        files = sorted((source / name).glob("*.npy"))
        if args.protocol == "random":
            if len(files) < 3:
                raise ValueError(f"Lớp {name} cần >=3 mẫu để chia ba tập")
            files = [files[i] for i in rng.permutation(len(files))]
            nv, nt = max(1, round(len(files) * args.val_ratio)), max(1, round(len(files) * args.test_ratio))
            if nv + nt >= len(files):
                raise ValueError(f"Lớp {name}: quá ít mẫu cho ratios đã chọn")
        for index, path in enumerate(files):
            signer = path.stem.split("_")[0]
            observed.add(signer)
            if args.protocol == "signer":
                if signer not in mapping:
                    raise ValueError(f"Signer {signer} chưa được khai báo; không tự bỏ mẫu")
                split = mapping[signer]
            else:
                split = "val" if index < nv else "test" if index < nv + nt else "train"
            rows.append({"path": os.path.relpath(path, output), "label": label, "split": split, "signer": signer})
    if args.protocol == "signer" and set(mapping) - observed:
        raise ValueError(f"Signer khai báo không tồn tại: {set(mapping) - observed}")
    if {row["split"] for row in rows} != {"train", "val", "test"}:
        raise ValueError("Một split rỗng; kiểm tra danh sách signer")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "split", "signer"])
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "classes.json", classes)
    write_json(output / "split_info.json", vars(args))
    print(f"{len(classes)} lớp, {len(rows)} mẫu. Đặt model.num_classes={len(classes)} trong config.")


if __name__ == "__main__":
    main()
