"""Sinh dữ liệu GIẢ để kiểm tra pipeline, không phải benchmark nhận diện ký hiệu."""

import argparse
import csv
from pathlib import Path

import numpy as np

from hagct.engine.common import write_json


def create_demo(output, seed=42):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Không ghi đè dữ liệu: {output}")
    (output / "samples").mkdir(parents=True)
    rng = np.random.default_rng(seed)
    rows = []
    for split, per_class in (("train", 5), ("val", 2), ("test", 2)):
        for label in range(3):
            for index in range(per_class):
                length = int(rng.integers(8, 23))
                raw = np.zeros((length, 27, 3), dtype=np.float32)
                base = rng.normal(0, 0.05, (27, 2)).astype(np.float32)
                base[0] = (0, 0)
                base[1], base[2] = (-0.5, 0.3), (0.5, 0.3)
                raw[:, :, :2] = base[None] + np.array([0.5, 0.2], dtype=np.float32)
                phase = np.linspace(0, 2 * np.pi, length)
                raw[:, 7:, label % 2] += np.sin(phase * (label + 1))[:, None] * 0.15
                raw[:, :, 2] = 1.0
                raw[length // 2, 10, 2] = 0.0  # Thử nhánh nội suy confidence thấp.
                path = f"samples/{split}_{label}_{index}.npy"
                np.save(output / path, raw, allow_pickle=False)
                rows.append({"path": path, "label": label, "split": split, "signer": split + "_synthetic"})
    with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "split", "signer"])
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "classes.json", ["synthetic_0", "synthetic_1", "synthetic_2"])
    print(f"Đã tạo {len(rows)} mẫu GIẢ tại {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/demo")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    create_demo(args.output, args.seed)
