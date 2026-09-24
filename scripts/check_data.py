"""Kiểm tra TOÀN BỘ tensor trước khi train; tùy chọn kiểm tra bản sao byte."""

import argparse
import hashlib
from collections import Counter

from hagct.config import load_config
from hagct.data.dataset import SkeletonDataset, read_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--check-duplicates", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    rows = read_manifest(cfg.data.manifest, cfg.model.num_classes, cfg.data.signer_disjoint)
    print("Mẫu theo split:", dict(Counter(row["split"] for row in rows)))
    if args.check_duplicates:
        seen = {}
        for row in rows:
            digest = hashlib.sha256()
            with open(row["path"], "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            key = digest.hexdigest()
            if key in seen:
                raise ValueError(f"Hai file trùng byte: {seen[key]} và {row['path']}")
            seen[key] = row["path"]
    for split in ("train", "val", "test"):
        if not any(row["split"] == split for row in rows):
            print(f"{split}: không có trong manifest")
            continue
        dataset = SkeletonDataset(rows, cfg.data, cfg.augment, split, training=False)
        lengths = [int(dataset[i]["mask"].sum()) for i in range(len(dataset))]
        missing = sorted(set(range(cfg.model.num_classes)) - set(dataset.labels))
        print(f"{split}: {len(dataset)} mẫu, valid_frames={min(lengths)}..{max(lengths)}, "
              f"lớp thiếu={missing}")
    print("OK: tất cả mẫu đọc được, shape/giá trị/preprocessing hợp lệ.")


if __name__ == "__main__":
    main()
