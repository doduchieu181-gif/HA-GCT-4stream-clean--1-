"""Chạy các CLI thật nối tiếp; dữ liệu/checkpoint GIẢ được ghi vào output riêng."""

import argparse
import subprocess
import sys
from pathlib import Path

from hagct.config import Config
from hagct.engine.common import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/smoke")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Chọn output mới để không ghi đè: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cfg = Config.from_dict({
        "data": {"manifest": str(output / "data/manifest.csv"), "max_frames": 16, "signer_disjoint": True},
        "model": {"num_classes": 3, "d_model": 16, "num_heads": 4, "spatial_layers": 1,
                  "temporal_layers": 1, "dropout": 0, "drop_path": 0},
        "train": {"device": "cpu", "epochs": 2, "batch_size": 4, "accum_steps": 3,
                  "warmup_epochs": 0, "amp": False, "output_dir": str(output / "supervised")}})
    config = output / "config.json"
    write_json(config, cfg.to_dict())

    def run(*command):
        print("\nRUN:", " ".join(map(str, command)), flush=True)
        subprocess.run([sys.executable, *map(str, command)], check=True)

    run("-m", "scripts.make_demo_data", "--output", output / "data")
    run("-m", "scripts.check_data", "--config", config, "--check-duplicates")
    run("-m", "scripts.trace_flow", "--config", config)
    run("train.py", "--config", config, "--threads", 1, "--stop-after", 1)
    run("train.py", "--config", config, "--threads", 1, "--resume", output / "supervised/last.pt")
    run("evaluate.py", "--checkpoint", output / "supervised/best.pt", "--output", output / "evaluation",
        "--split", "test", "--device", "cpu", "--threads", 1)
    run("predict.py", "--checkpoint", output / "supervised/best.pt", "--input", output / "data/samples/test_0_0.npy",
        "--classes", output / "data/classes.json", "--device", "cpu", "--threads", 1)
    run("pretrain.py", "--config", config, "--output", output / "pretrain", "--epochs", 1, "--threads", 1)
    run("train.py", "--config", config, "--output", output / "finetune", "--epochs", 1, "--threads", 1,
        "--pretrained", output / "pretrain/best.pt")
    print("\nSMOKE TEST PASSED. Đây không phải đánh giá chất lượng mô hình trên dữ liệu thật.")


if __name__ == "__main__":
    main()
