"""Luồng chính: sinh 4 tín hiệu -> 4 encoder riêng -> cộng logits có trọng số."""

import torch
from torch import nn

from hagct.topology import PARENTS
from .encoder import ClassificationHead, HAGCTEncoder

STREAM_NAMES = ("joint", "bone", "motion", "bone_motion")


def build_streams(x, mask, parents=None):
    if x.ndim != 4 or x.shape[1] != 2 or x.shape[3] != 27:
        raise ValueError("build_streams cần x=(B,2,T,27)")
    if mask.shape != (x.shape[0], x.shape[2]) or mask.dtype != torch.bool:
        raise ValueError("Mask phải bool (B,T)")
    if parents is None:
        parents = torch.as_tensor(PARENTS, device=x.device)
    joint = x.masked_fill(~mask[:, None, :, None], 0)
    bone = joint - joint.index_select(3, parents)

    def difference(sequence):
        velocity = torch.zeros_like(sequence)
        adjacent = mask[:, 1:] & mask[:, :-1]
        velocity[:, :, 1:] = (sequence[:, :, 1:] - sequence[:, :, :-1]).masked_fill(
            ~adjacent[:, None, :, None], 0)
        return velocity

    return {"joint": joint, "bone": bone,
            "motion": difference(joint), "bone_motion": difference(bone)}


class FourStreamHAGCT(nn.Module):
    def __init__(self, cfg, max_frames):
        super().__init__()
        self.register_buffer("parents", torch.as_tensor(PARENTS))
        self.encoders = nn.ModuleDict({name: HAGCTEncoder(cfg, max_frames) for name in STREAM_NAMES})
        self.heads = nn.ModuleDict({name: ClassificationHead(cfg.d_model, cfg.num_classes, cfg.dropout)
                                    for name in STREAM_NAMES})
        self.fusion_logits = nn.Parameter(torch.zeros(4))  # Khởi đầu: 0.25 mỗi stream.

    def fusion_weights(self):
        return self.fusion_logits.softmax(dim=0)

    def forward(self, x, mask, return_streams=False):
        streams = build_streams(x, mask, self.parents)
        logits = {name: self.heads[name](self.encoders[name](streams[name], mask))
                  for name in STREAM_NAMES}
        stacked = torch.stack([logits[name] for name in STREAM_NAMES], dim=0)
        fused = (stacked * self.fusion_weights()[:, None, None]).sum(dim=0)
        return (fused, logits) if return_streams else fused

    def initialize_encoders(self, state):
        """Nạp cùng encoder pretrained vào CẢ BỐN stream; sai shape phải báo lỗi."""
        expected = self.encoders["joint"].state_dict()
        if set(state) != set(expected):
            raise ValueError("Pretrained encoder thiếu/thừa key; không tương thích bản clean")
        bad = [key for key in expected if expected[key].shape != state[key].shape]
        if bad:
            raise ValueError(f"Pretrained sai kích thước: {bad}")
        for encoder in self.encoders.values():
            encoder.load_state_dict(state, strict=True)
        return {name: len(state) for name in STREAM_NAMES}
