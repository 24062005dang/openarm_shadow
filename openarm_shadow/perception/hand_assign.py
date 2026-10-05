"""Gán bàn tay MediaPipe cho cổ tay Pose (không tin nhãn handedness)."""
from __future__ import annotations

import numpy as np


def assign_hands_to_wrists(hand_roots, wrists, gate):
    """hand_roots: [(x, y)] cổ tay của từng bàn tay MediaPipe; wrists: {side: (x, y) hoặc None}; gate: khoảng
    cách tối đa (cùng đơn vị). Trả list side|None theo thứ tự hand_roots, mỗi side dùng tối đa 1 lần.
    Ghép cặp theo khoảng cách tăng dần (với 2 tay x 2 cổ tay, kết quả trùng tối ưu tổng khoảng cách)."""
    pairs = []
    for i, r in enumerate(hand_roots):
        for side, wr in wrists.items():
            if wr is None or not np.all(np.isfinite(wr)):
                continue
            d = float(np.linalg.norm(np.asarray(r, float) - np.asarray(wr, float)))
            if d < gate:
                pairs.append((d, i, side))
    out, used = [None] * len(hand_roots), set()
    for d, i, side in sorted(pairs):
        if out[i] is None and side not in used:
            out[i], used = side, used | {side}
    return out
