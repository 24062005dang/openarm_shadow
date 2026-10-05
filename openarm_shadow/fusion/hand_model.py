"""Làm sạch bàn tay 3D sau triangulation: lọc đốt xương bất thường (HandShape), hướng lòng bàn tay bằng Kabsch
với khuôn học từ người dùng (PalmModel), kiểm tra giải phẫu bàn tay - cẳng tay."""
from __future__ import annotations

from collections import deque

import numpy as np

from ..core.geometry import unit
from ..perception.hand_geometry import palm_frame_from_depth

# Chỉ số MediaPipe Hand
WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP = 0, 5, 9, 13, 17
PALM = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
# 20 đốt: (gốc, đầu)
BONES = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20))


def hand_forearm_angle(R_hand, elbow, wrist):
    """Góc (độ) giữa hướng ngón tay (cột x của khung bàn tay) và cẳng tay (khuỷu -> cổ tay)."""
    if R_hand is None or elbow is None or wrist is None:
        return np.nan
    f = np.asarray(wrist, float) - np.asarray(elbow, float)
    if not np.all(np.isfinite(f)) or np.linalg.norm(f) < 1e-6:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(unit(f) @ unit(R_hand[:, 0]), -1.0, 1.0))))


class HandShape:
    """Chiều dài đốt xương bàn tay (học trung vị) và lọc điểm có đốt dài/ngắn bất thường."""

    def __init__(self, min_ratio=0.6, max_ratio=1.6, samples=60, min_samples=10):
        self.min_ratio, self.max_ratio, self.min_samples = min_ratio, max_ratio, min_samples
        self.hist = [deque(maxlen=samples) for _ in BONES]

    def lengths(self):
        return np.array([np.median(h) if len(h) >= self.min_samples else np.nan for h in self.hist])

    def filter(self, pts, learn=True):
        """pts (21,3) có thể NaN. Trả (pts_sạch, số điểm bị bỏ). learn: thêm khung này vào thống kê
        (chỉ các đốt đạt kiểm tra, để một khung sai không làm hỏng chiều dài chuẩn)."""
        p = np.array(pts, float)
        ref = self.lengths()
        dropped = 0
        for k, (a, b) in enumerate(BONES):
            if not (np.all(np.isfinite(p[a])) and np.all(np.isfinite(p[b]))):
                continue
            L = float(np.linalg.norm(p[b] - p[a]))
            if np.isfinite(ref[k]) and not (self.min_ratio * ref[k] <= L <= self.max_ratio * ref[k]):
                p[b] = np.nan
                dropped += 1
                continue
            if learn and 0.005 < L < 0.15:
                self.hist[k].append(L)
        return p, dropped


class PalmModel:
    """Hướng lòng bàn tay từ 5 điểm cứng bằng Kabsch có trọng số với khuôn học được.

    Khung trả về cùng quy ước palm_frame_from_depth (x hướng ngón, z pháp tuyến ra khỏi lòng bàn tay), nên thay
    thế trực tiếp được. Trước khi học xong khuôn (learn_frames khung tốt), dùng palm_frame_from_depth.
    """

    def __init__(self, side, learn_frames=15, max_rms_m=0.012, outlier_m=0.015):
        self.side, self.learn_frames = side, learn_frames
        self.max_rms, self.outlier = max_rms_m, outlier_m
        self._samples = []
        self.template = None           # (5,3) toạ độ trong khung lòng bàn tay, gốc = tâm

    def _learn(self, pts, R, center):
        local = (pts[list(PALM)] - center) @ R
        self._samples.append(local)
        if len(self._samples) >= self.learn_frames:
            T = np.median(np.array(self._samples), axis=0)
            self.template = T - T.mean(axis=0)
            self._samples = []

    def estimate(self, pts, weights=None, quality=1.0):
        """-> (R, center, rms_m, mode). mode: "KABSCH", "3PT" (chưa có khuôn) hoặc lý do bị loại."""
        p = np.asarray(pts, float)
        if self.template is None:
            R, center = palm_frame_from_depth(p, side=self.side)
            if R is None:
                return None, None, np.nan, "MISSING"
            # Chỉ học từ khung thấy rõ lòng bàn tay (quality cao) để khuôn không mang dấu lật
            if quality >= 0.5:
                self._learn(p, R, center)
            return R, center, np.nan, "3PT"
        idx = [k for k, j in enumerate(PALM) if np.all(np.isfinite(p[j]))]
        if len(idx) < 3 or 0 not in idx:
            return None, None, np.nan, "MISSING"
        w = np.ones(len(PALM)) if weights is None else np.asarray(weights, float)
        while True:
            A = self.template[idx]
            B = p[[PALM[k] for k in idx]]
            ww = np.clip(w[idx], 1e-3, None)
            ca = (ww[:, None] * A).sum(0) / ww.sum()
            cb = (ww[:, None] * B).sum(0) / ww.sum()
            Hm = ((A - ca) * ww[:, None]).T @ (B - cb)
            U, _, Vt = np.linalg.svd(Hm)
            D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
            R = Vt.T @ D @ U.T                           # B ~ R A: cột của R = trục lòng bàn tay trong world
            res = np.linalg.norm((A - ca) @ R.T + cb - B, axis=1)
            worst = int(np.argmax(res))
            if res[worst] > self.outlier and len(idx) > 3 and idx[worst] != 0:
                idx.pop(worst)
                continue
            break
        rms = float(np.sqrt(np.mean(res ** 2)))
        if rms > self.max_rms:
            return None, None, rms, "BAD-FIT"
        center = cb - R @ ca
        return R, center, rms, "KABSCH"
