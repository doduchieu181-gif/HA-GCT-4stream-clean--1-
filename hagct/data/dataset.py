"""Manifest CSV rõ ràng thay vì đoán format hoặc tự chia lại tập khi train."""

import csv
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .preprocess import preprocess, pad_sequence
from .augment import augment_and_pad


def read_manifest(path, num_classes, signer_disjoint=False):
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"path", "label", "split"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV cần các cột path,label,split; xem docs/02_DATA.md")
        rows = list(reader)
    seen, signers = set(), {}
    for row in rows:
        file = (path.parent / row["path"]).resolve()
        if not file.is_file():
            raise FileNotFoundError(file)
        if file in seen:
            raise ValueError(f"File xuất hiện nhiều lần (nguy cơ rò rỉ split): {file}")
        seen.add(file)
        row["path"] = str(file)
        row["label"] = int(row["label"])
        if not 0 <= row["label"] < num_classes:
            raise ValueError(f"label phải trong [0,{num_classes-1}]: {row}")
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError("split chỉ gồm train/val/test")
        row["length"] = int(row["length"]) if row.get("length", "").strip() else None
        if signer_disjoint:
            signer = row.get("signer", "").strip()
            if not signer:
                raise ValueError("signer_disjoint cần cột signer ở mọi dòng")
            if signer in signers and signers[signer] != row["split"]:
                raise ValueError(f"Signer {signer} bị trùng giữa các split")
            signers[signer] = row["split"]
    if not rows:
        raise ValueError("Manifest rỗng")
    return rows


def manifest_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SkeletonDataset(Dataset):
    def __init__(self, rows, data_cfg, augment_cfg, split, seed=42, training=None):
        self.rows = [r for r in rows if r["split"] == split]
        if not self.rows:
            raise ValueError(f"Không có mẫu cho split={split}")
        self.data_cfg, self.augment_cfg = data_cfg, augment_cfg
        self.training = (split == "train") if training is None else training
        self.seed, self.epoch = seed, 0
        self.labels = [r["label"] for r in self.rows]
        self.cache = [self._read(i) for i in range(len(self))] if data_cfg.cache else None

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _read(self, index):
        row = self.rows[index]
        try:
            return preprocess(np.load(row["path"], allow_pickle=False), self.data_cfg, row["length"])
        except Exception as error:
            raise ValueError(f"Lỗi mẫu {row['path']}: {error}") from error

    def __getitem__(self, index):
        x = self.cache[index] if self.cache is not None else self._read(index)
        if self.training:
            # Tái lập được qua epoch/resume và không phụ thuộc số DataLoader worker.
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
            x, mask = augment_and_pad(x, self.augment_cfg, self.data_cfg.max_frames, rng)
        else:
            x, mask = pad_sequence(x, self.data_cfg.max_frames)
        return {"x": torch.from_numpy(x), "mask": torch.from_numpy(mask),
                "label": torch.tensor(self.labels[index], dtype=torch.long), "index": index}
