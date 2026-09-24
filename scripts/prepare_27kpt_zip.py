"""Chuẩn bị trực tiếp gói 27kpt.zip được chia sẻ cùng MultiVSL200.

Gói đã có skeleton (T,27,3) và labels.csv, nên không cần chạy MediaPipe.
Mặc định chỉ lấy 10 lớp đầu để kiểm tra nhanh toàn bộ pipeline.
"""

import argparse
import csv
import io
import json
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np

from hagct.config import Config
from hagct.engine.common import write_json


SIGNER_PATTERN = re.compile(r"signer(\d+)")


def prepare(archive, output, num_classes=10,
            val_signers=("02", "05"), test_signers=("03", "06")):
    archive, output = Path(archive).resolve(), Path(output).resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output đã có dữ liệu: {output}")
    if not 2 <= num_classes <= 199:
        raise ValueError("num-classes phải trong [2,199]")
    val_signers, test_signers = set(val_signers), set(test_signers)
    if val_signers & test_signers:
        raise ValueError("Signer val và test không được trùng")

    with zipfile.ZipFile(archive) as source:
        try:
            labels_text = io.TextIOWrapper(source.open("raw_npy/labels.csv"), encoding="utf-8")
        except KeyError as error:
            raise ValueError("ZIP không có raw_npy/labels.csv") from error
        annotations = list(csv.reader(labels_text))
        if not annotations or any(len(row) != 2 for row in annotations):
            raise ValueError("labels.csv không đúng dạng: sample_name,label")
        labels = sorted({int(row[1]) for row in annotations})
        selected_labels = labels[:num_classes]
        remap = {original: new for new, original in enumerate(selected_labels)}

        members = {Path(name).stem: name for name in source.namelist() if name.endswith(".npy")}
        prepared = []
        seen = set()
        for stem, raw_label in annotations:
            original_label = int(raw_label)
            if original_label not in remap:
                continue
            if stem in seen:
                raise ValueError(f"Tên sample bị lặp trong labels.csv: {stem}")
            seen.add(stem)
            match = SIGNER_PATTERN.search(stem)
            if not match or stem not in members:
                raise ValueError(f"Thiếu signer hoặc file npy cho sample: {stem}")
            signer = match.group(1)
            split = "val" if signer in val_signers else "test" if signer in test_signers else "train"
            prepared.append((stem, remap[original_label], split, signer, original_label, members[stem]))

        if not prepared:
            raise ValueError("Không có sample phù hợp")
        output.mkdir(parents=True, exist_ok=True)
        samples = output / "samples"
        samples.mkdir()
        rows = []
        for index, (stem, label, split, signer, original, member) in enumerate(prepared, 1):
            destination = samples / f"{stem}.npy"
            # Không dùng extract() để tên trong ZIP không thể thoát khỏi output.
            with source.open(member) as input_file, destination.open("wb") as output_file:
                shutil.copyfileobj(input_file, output_file)
            value = np.load(destination, allow_pickle=False)
            if value.shape != (150, 27, 3) or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
                raise ValueError(f"Sample không đúng (150,27,3) hữu hạn: {stem} {value.shape}")
            rows.append({"path": f"samples/{destination.name}", "label": label, "split": split,
                         "signer": signer, "original_label": original})
            if index % 100 == 0:
                print(f"Đã chuẩn bị {index}/{len(prepared)} sample", flush=True)

    for split in ("train", "val", "test"):
        subset = [row for row in rows if row["split"] == split]
        if not subset:
            raise ValueError(f"Split {split} rỗng; đổi danh sách signer")
        present = {row["label"] for row in subset}
        if split == "train" and present != set(range(num_classes)):
            raise ValueError(f"Train thiếu lớp: {sorted(set(range(num_classes)) - present)}")

    with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "classes.json", [f"original_label_{value:03d}" for value in selected_labels])
    write_json(output / "split_info.json", {
        "source_archive": str(archive), "selected_original_labels": selected_labels,
        "label_mapping": {str(old): new for old, new in remap.items()},
        "val_signers": sorted(val_signers), "test_signers": sorted(test_signers),
        "ignored_unlabelled_npy": len(members) - len(annotations),
        "counts": {split: sum(row["split"] == split for row in rows)
                   for split in ("train", "val", "test")},
    })

    cfg = Config.from_dict({
        "data": {"manifest": str(output / "manifest.csv"), "layout": "TVC",
                 "channels": "xyz", "max_frames": 150, "filter_enabled": True,
                 "signer_disjoint": True},
        "model": {"num_classes": num_classes, "d_model": 32, "spatial_layers": 1,
                  "temporal_layers": 1, "num_heads": 4, "dropout": 0.1, "drop_path": 0.0},
        "train": {"device": "auto", "batch_size": 4, "epochs": 2, "accum_steps": 2,
                  "amp": True, "warmup_epochs": 0, "patience": 5,
                  "output_dir": str(Path("runs") / output.name)},
    })
    write_json(output / "trial_config.json", cfg.to_dict())
    print("\nHOÀN TẤT")
    print(f"Mẫu: {len(rows)} | lớp: {num_classes} | output: {output}")
    print(f"Config chạy thử: {output / 'trial_config.json'}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, help="Đường dẫn 27kpt.zip")
    parser.add_argument("--output", default="data/trial_27kpt")
    parser.add_argument("--num-classes", type=int, default=10,
                        help="10 để thử nhanh; 199 để chuẩn bị toàn bộ")
    parser.add_argument("--val-signers", nargs="+", default=["02", "05"])
    parser.add_argument("--test-signers", nargs="+", default=["03", "06"])
    args = parser.parse_args()
    prepare(args.archive, args.output, args.num_classes, args.val_signers, args.test_signers)


if __name__ == "__main__":
    main()
