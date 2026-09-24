"""In từng bước và tensor shape của MỘT mẫu để học luồng chạy."""

import argparse

import numpy as np
import torch

from hagct.config import load_config
from hagct.data.dataset import SkeletonDataset, read_manifest
from hagct.models import FourStreamHAGCT, build_streams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = load_config(args.config)
    rows = read_manifest(cfg.data.manifest, cfg.model.num_classes, cfg.data.signer_disjoint)
    dataset = SkeletonDataset(rows, cfg.data, cfg.augment, "train", training=False)
    row = dataset.rows[0]
    raw = np.load(row["path"], allow_pickle=False)
    sample = dataset[0]
    x, mask = sample["x"][None], sample["mask"][None]
    model = FourStreamHAGCT(cfg.model, cfg.data.max_frames).eval()
    print("Đây là model KHỞI TẠO NGẪU NHIÊN, chỉ minh họa shape, không phải dự đoán đã học.")
    print(f"1. manifest -> {row['path']} | raw={raw.shape}, layout={cfg.data.layout}, channels={cfg.data.channels}")
    print(f"2. preprocess + pad -> x={tuple(x.shape)} | mask={tuple(mask.shape)}, valid={int(mask.sum())}")
    with torch.inference_mode():
        streams = build_streams(x, mask)
        scores = []
        for name, signal in streams.items():
            features = model.encoders[name](signal, mask)
            logits = model.heads[name](features)
            print(f"3. {name:11s}: {tuple(signal.shape)} -> encoder {tuple(features.shape)} -> head {tuple(logits.shape)}")
            scores.append(logits)
        fused = (torch.stack(scores) * model.fusion_weights()[:, None, None]).sum(0)
        print(f"4. Softmax fusion weights={model.fusion_weights().tolist()} -> logits={tuple(fused.shape)}")
        print("5. Training: CE/Focal(logits, label) -> backward -> optimizer; prediction: softmax -> top-k")
        print(f"Tổng tham số bốn stream: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    main()
