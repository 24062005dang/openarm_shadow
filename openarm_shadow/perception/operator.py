"""Khoá người điều khiển trong các người Pose thấy được ở một camera."""
from __future__ import annotations

import numpy as np

from .types import L_EL, L_HIP, L_SH, NOSE, R_EL, R_HIP, R_SH


def shoulders(pose_2d):
    """(tâm 2 vai, độ rộng vai) theo toạ độ ảnh chuẩn hoá, hoặc None nếu vai không thấy."""
    a, b = pose_2d[L_SH], pose_2d[R_SH]
    if min(a[2], b[2]) < 0.3:
        return None
    return 0.5 * (a[:2] + b[:2]), float(np.linalg.norm(a[:2] - b[:2]))


def select_operator(cands, prev, lock_dist=1.0):
    """Chọn người điều khiển trong các người Pose thấy (không nhảy sang người khác trong khung).

    - Đang khoá (prev = pose người điều khiển lần trước): chọn người có tâm 2 vai gần tâm cũ nhất, chỉ nhận nếu cách
      tâm cũ < lock_dist x độ rộng vai cũ và độ rộng vai không đổi quá 2 lần. Không ai thoả -> None (coi như hụt:
      giữ ngắn rồi mất người, robot đứng yên) thay vì nhận người khác.
    - Chưa khoá / vừa mất người: chọn người TO nhất (vai rộng nhất = đứng gần camera nhất), ưu tiên gần giữa ảnh và
      thấy đủ người (mũi, khuỷu, hông): người ngồi sau bàn / bị cắt nửa người bị ưu tiên thấp hơn.
    """
    info = [(shoulders(c), k) for k, c in enumerate(cands)]
    info = [(s, k) for s, k in info if s is not None and s[1] > 1e-3]
    if not info:
        return None
    ps = shoulders(prev) if prev is not None else None
    if ps is not None and ps[1] > 1e-3:
        best = min(info, key=lambda x: np.linalg.norm(x[0][0] - ps[0]))
        (c, w), k = best
        if np.linalg.norm(c - ps[0]) < lock_dist * ps[1] and 0.5 < w / ps[1] < 2.0:
            return k
        return None
    def score(x):
        (c, w), k = x
        center = 1.0 - 0.5 * min(1.0, abs(c[0] - 0.5) * 2)
        full = float(np.mean(np.clip(cands[k][[NOSE, L_EL, R_EL, L_HIP, R_HIP], 2], 0, 1)))
        return w * center * (0.3 + 0.7 * full)
    return max(info, key=score)[1]
