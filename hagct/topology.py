"""Một nguồn DUY NHẤT cho thứ tự 27 joint, dùng ở graph, bone và augmentation.

Thứ tự này tương ứng sign_27 trong nhánh hagct, không phải MediaPipe 33/21 gốc.
Phải chuyển dữ liệu về đúng thứ tự trước khi train.
"""

import numpy as np

JOINT_NAMES = [
    "nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_palm", "left_thumb", "left_index_root",
    "left_index_tip", "left_middle_root", "left_middle_tip", "left_ring_root",
    "left_ring_tip", "left_pinky_root", "left_pinky_tip", "right_palm",
    "right_thumb", "right_index_root", "right_index_tip", "right_middle_root",
    "right_middle_tip", "right_ring_root", "right_ring_tip", "right_pinky_root",
    "right_pinky_tip",
]
PARENTS = np.array([0, 0, 0, 1, 2, 3, 4, 5, 7, 7, 9, 7, 11, 7, 13, 7, 15,
                    6, 17, 17, 19, 17, 21, 17, 23, 17, 25], dtype=np.int64)
NUM_JOINTS = 27


def graph_subsets():
    """Trả 3 ma trận (self, inward, outward), A[nguồn, đích]."""
    eye = np.eye(NUM_JOINTS, dtype=np.float32)
    inward = np.zeros_like(eye)
    hands = np.zeros_like(eye)
    for child in range(1, NUM_JOINTS):
        parent = PARENTS[child]
        inward[parent, child] = 1
        if (7 <= child <= 16 and 7 <= parent <= 16) or (17 <= child and 17 <= parent):
            hands[parent, child] = 1
    hands[7, 17] = 1   # Tương tác giữa hai bàn tay
    body = np.stack([eye, inward, inward.T])
    hand = np.stack([eye, hands, hands.T])
    return body, hand
