"""Đổi tensor nhiều mẫu sang manifest. Không đoán chiều hoặc valid_length."""

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np


def read_labels(path, trust_pickle):
    path = Path(path)
    if path.suffix == ".json":
        labels = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix == ".npy":
        labels = np.load(path, allow_pickle=False)
    elif path.suffix in {".pkl", ".pickle"}:
        if not trust_pickle:
            raise ValueError("Pickle có thể chạy code. Chỉ dùng --trust-pickle với file do bạn tin cậy.")
        with path.open("rb") as handle:
            value = pickle.load(handle)
        # Dạng legacy: (sample_names, labels). List nhãn số cũng được hỗ trợ.
        labels = value[1] if isinstance(value, tuple) and len(value) == 2 else value
    else:
        raise ValueError("Labels cần .json, .npy hoặc .pkl tin cậy")
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.dtype.kind not in "iu":
        raise ValueError("Labels phải vector số nguyên; không tự đoán/remap nhãn")
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--layout", required=True, choices=["NCTV", "NTVC", "NCTVM"])
    parser.add_argument("--channels", required=True, choices=["xy", "xy_conf", "xyz"])
    parser.add_argument("--person-index", type=int, help="Bắt buộc với NCTVM; không tự trộn người")
    parser.add_argument("--lengths", help=".npy vector số frame thật (right-padding ở cuối)")
    parser.add_argument("--all-frames-valid", action="store_true", help="Xác nhận file không chứa padding")
    parser.add_argument("--num-classes", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trust-pickle", action="store_true")
    args = parser.parse_args()
    if bool(args.lengths) == bool(args.all_frames_valid):
        raise ValueError("Chọn --lengths HOẶC --all-frames-valid; không đoán frame đệm")
    output = Path(args.output)
    split_dir = output / args.split
    if split_dir.exists():
        raise FileExistsError(f"Split đã tồn tại: {split_dir}")
    labels = read_labels(args.labels, args.trust_pickle)
    data = np.load(args.data, mmap_mode="r", allow_pickle=False)
    if args.layout == "NCTVM":
        if data.ndim != 5 or args.person_index is None or not 0 <= args.person_index < data.shape[-1]:
            raise ValueError("NCTVM cần data 5D và person-index hợp lệ")
        data = data[..., args.person_index]
    elif args.person_index is not None:
        raise ValueError("person-index chỉ dùng với NCTVM")
    if data.ndim != 4 or len(data) != len(labels):
        raise ValueError("Data phải 4D sau chọn người, N phải bằng số labels")
    if args.layout != "NTVC":
        data = data.transpose(0, 2, 3, 1)  # Ghi đầu ra luôn NTVC => từng file TVC.
    expected_channels = 2 if args.channels == "xy" else 3
    if data.shape[2:] != (27, expected_channels):
        raise ValueError(f"Cần 27 joints và {expected_channels} channels, nhận {data.shape}")
    if not len(labels) or args.num_classes < 1 or labels.min() < 0 or labels.max() >= args.num_classes:
        raise ValueError("Nhãn phải là 0..num_classes-1; không tự trừ 1")
    lengths = (np.load(args.lengths, allow_pickle=False) if args.lengths else
               np.full(len(data), data.shape[1], dtype=np.int64))
    if lengths.shape != (len(data),) or lengths.dtype.kind not in "iu" or np.any(lengths < 1) or np.any(lengths > data.shape[1]):
        raise ValueError("lengths phải là vector nguyên trong [1,T]")
    # Nối các split nhưng phải cùng semantic channels và số lớp.
    info = {"channels": args.channels, "num_classes": args.num_classes, "output_layout": "TVC"}
    metadata = output / "conversion.json"
    if (output / "manifest.csv").exists() and not metadata.exists():
        raise ValueError("Output có manifest không do converter này tạo; chọn output mới")
    if metadata.exists() and json.loads(metadata.read_text()) != info:
        raise ValueError("Các split không cùng cấu hình conversion")
    split_dir.mkdir(parents=True)
    manifest = output / "manifest.csv"
    new_manifest = not manifest.exists()
    with manifest.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if new_manifest:
            writer.writerow(["path", "label", "split"])
        for index, (sample, label, length) in enumerate(zip(data, labels, lengths)):
            relative = f"{args.split}/{index:07d}.npy"
            np.save(output / relative, np.asarray(sample[:length], dtype=np.float32), allow_pickle=False)
            writer.writerow([relative, int(label), args.split])
    metadata.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"Đã ghi {len(labels)} mẫu {args.split}; config: layout=TVC, channels={args.channels}.")


if __name__ == "__main__":
    main()
