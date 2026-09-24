"""Đọc JSON và kiểm tra cấu hình sớm, trước khi tải dữ liệu lên GPU."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    manifest: str = "data/manifest.csv"
    layout: str = "TVC"       # TVC=(frame, joint, channel); CTV=(channel, frame, joint)
    channels: str = "xy_conf"  # xy, xy_conf hoặc xyz; KHÔNG đoán Z là confidence
    max_frames: int = 150
    fps: float = 30.0
    confidence_threshold: float = 0.3
    zero_is_missing: bool = False
    root_joint: int = 0
    scale_normalization: bool = True
    filter_enabled: bool = True
    min_cutoff: float = 1.0
    beta: float = 0.7
    derivative_cutoff: float = 1.0
    cache: bool = False
    signer_disjoint: bool = False


@dataclass
class AugmentConfig:
    enabled: bool = True
    crop_probability: float = 0.5
    crop_min_ratio: float = 0.8
    speed_probability: float = 0.5
    speed_min: float = 0.85
    speed_max: float = 1.15
    rotation_degrees: float = 15.0
    bone_scale: float = 0.05
    mask_probability: float = 0.3
    num_mask_joints: int = 2
    noise_std: float = 0.01
    random_placement: bool = True


@dataclass
class ModelConfig:
    num_classes: int = 199
    d_model: int = 128
    spatial_layers: int = 3
    temporal_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    drop_path: float = 0.1


@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "auto"
    batch_size: int = 8
    num_workers: int = 0      # Chạy được trên Windows; tăng sau khi kiểm tra RAM
    epochs: int = 100
    lr: float = 0.0003
    classifier_lr_multiplier: float = 1.0
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    min_lr_ratio: float = 0.01
    accum_steps: int = 4
    amp: bool = True
    clip_grad: float = 1.0
    label_smoothing: float = 0.1
    loss: str = "ce"
    focal_gamma: float = 2.0
    balanced_sampling: bool = False
    mixup_alpha: float = 0.0  # Bật sau baseline; trộn CHỈ các cặp cùng mask
    patience: int = 20
    output_dir: str = "runs/baseline"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        unknown = set(value) - {"data", "augment", "model", "train"}
        if unknown:
            raise ValueError(f"Nhóm cấu hình không hợp lệ: {unknown}")
        cfg = cls(**{name: kind(**value.get(name, {})) for name, kind in (
            ("data", DataConfig), ("augment", AugmentConfig),
            ("model", ModelConfig), ("train", TrainConfig))})
        cfg.validate()
        return cfg

    def validate(self):
        d, a, m, t = self.data, self.augment, self.model, self.train
        if d.layout not in {"TVC", "CTV"} or d.channels not in {"xy", "xy_conf", "xyz"}:
            raise ValueError("layout phải là TVC/CTV; channels phải là xy/xy_conf/xyz")
        if d.max_frames < 2 or d.fps <= 0 or not 0 <= d.root_joint < 27:
            raise ValueError("max_frames >= 2, fps > 0, root_joint thuộc [0,26]")
        if min(d.min_cutoff, d.derivative_cutoff) <= 0 or d.beta < 0:
            raise ValueError("Cutoff phải dương, beta không âm")
        if not 0 <= d.confidence_threshold <= 1:
            raise ValueError("confidence_threshold thuộc [0,1]")
        if min(m.num_classes, m.d_model, m.num_heads, m.spatial_layers, m.temporal_layers) < 1:
            raise ValueError("Kích thước mô hình phải dương")
        if m.d_model % m.num_heads or m.d_model < 4:
            raise ValueError("d_model >= 4 và chia hết cho num_heads")
        if not 0 <= m.dropout < 1 or not 0 <= m.drop_path < 1:
            raise ValueError("dropout/drop_path thuộc [0,1)")
        if min(t.batch_size, t.epochs, t.accum_steps, t.patience) < 1 or t.num_workers < 0 or t.seed < 0:
            raise ValueError("batch/epochs/accum/patience phải dương; workers không âm")
        if t.lr <= 0 or t.classifier_lr_multiplier <= 0 or t.weight_decay < 0 or t.clip_grad <= 0:
            raise ValueError("Tham số optimizer không hợp lệ")
        if t.warmup_epochs < 0 or not 0 <= t.min_lr_ratio <= 1:
            raise ValueError("Warmup/LR ratio không hợp lệ")
        if t.loss not in {"ce", "focal"} or not 0 <= t.label_smoothing < 1:
            raise ValueError("loss=ce/focal; label_smoothing thuộc [0,1)")
        if t.mixup_alpha < 0 or t.focal_gamma < 0:
            raise ValueError("mixup_alpha/focal_gamma không âm")
        for p in (a.crop_probability, a.speed_probability, a.mask_probability):
            if not 0 <= p <= 1:
                raise ValueError("Xác suất augmentation thuộc [0,1]")
        if not 0 < a.crop_min_ratio <= 1 or not 0 < a.speed_min <= a.speed_max:
            raise ValueError("Crop/speed không hợp lệ")
        if not 0 <= a.bone_scale < 1 or not 0 <= a.num_mask_joints <= 26:
            raise ValueError("bone_scale thuộc [0,1); num_mask_joints thuộc [0,26]")
        if a.noise_std < 0 or a.rotation_degrees < 0:
            raise ValueError("Noise/rotation không âm")


def load_config(path):
    return Config.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
