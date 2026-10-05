"""Khung thân người từ điểm Pose (x trước, y trái, z lên)."""
from __future__ import annotations

import numpy as np

from ..core.geometry import make_frame
from .types import L_HIP, L_SH, R_HIP, R_SH


def body_frame(W, vis, min_hip_vis=0.5):
    """Khung thân từ điểm world của Pose. Thiếu hông (ngồi, bị bàn che) thì dùng 'lên' của camera."""
    if min(vis[L_HIP], vis[R_HIP]) >= min_hip_vis:
        bottom = 0.5 * (W[L_HIP] + W[R_HIP])
    else:
        bottom = 0.5 * (W[L_SH] + W[R_SH]) + np.array([0.0, 0.5, 0.0])   # y của MediaPipe hướng xuống
    return make_frame(W[L_SH], W[R_SH], bottom)
