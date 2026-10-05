"""Lọc điểm 3D vai/khuỷu/cổ tay trước retarget.

- EMA điểm mốc (Hand Shadowing, arXiv 2603.11383, a = 0.8).
- ArmShape: đoạn tay dài/ngắn bất thường so với độ dài đã học -> không tin đoạn đó.
- PointKalman (tuỳ chọn): Kalman vận tốc không đổi, nhiễu đo theo độ tin cậy.
"""
from __future__ import annotations

import numpy as np


class EMA:
    """Làm mượt mũ đơn giản: y = a·x + (1−a)·y_prev (Hand Shadowing dùng a = 0.8 cho điểm mốc)."""

    def __init__(self, alpha):
        self.alpha = alpha
        self.y = None

    def __call__(self, x):
        x = np.asarray(x, float)
        self.y = x if self.y is None or x.shape != self.y.shape else self.alpha * x + (1 - self.alpha) * self.y
        return self.y

    def reset(self):
        self.y = None


class ArmShape:
    """Lọc khung xương cánh tay: độ dài cánh tay trên (vai -> khuỷu) và cẳng tay (khuỷu -> cổ tay) của người điều
    khiển không đổi. Học trung vị từ các khung tốt; khung có đoạn lệch quá `tol` (tỉ lệ) -> đoạn đó không tin được
    (điểm khuỷu/cổ tay sai, vd depth rơi vào thân làm khuỷu nhảy trước/sau thân -> J1/J3 lật).

    Chỉ dùng khi điểm có thang đo mét nhất quán giữa các khung (fusion triangulate). Khung sai không vào thống kê;
    mất người lâu hơn reset_s -> học lại (có thể là người khác)."""

    def __init__(self, tol=0.25, samples=90, min_samples=15, reset_s=2.0, len_range=(0.12, 0.5)):
        from collections import deque
        self.tol, self.min_samples, self.reset_s = float(tol), int(min_samples), float(reset_s)
        self.len_range = tuple(float(v) for v in len_range)
        self.hist = [deque(maxlen=int(samples)), deque(maxlen=int(samples))]   # cánh tay trên, cẳng tay
        self.t_last = None

    def reset(self):
        for h in self.hist:
            h.clear()

    def ref(self):
        return [float(np.median(h)) if len(h) >= self.min_samples else np.nan for h in self.hist]

    def __call__(self, s, e, w, t, learn=(True, True)):
        """-> (ok_upper, ok_fore, info). learn[k]: đoạn k đủ tin cậy để học (theo độ tin cậy nhận diện)."""
        if self.t_last is not None and t - self.t_last > self.reset_s:
            self.reset()
        self.t_last = t
        lengths = [float(np.linalg.norm(np.asarray(e) - s)), float(np.linalg.norm(np.asarray(w) - e))]
        ref = self.ref()
        ok, ratio = [], []
        lo, hi = self.len_range
        for k in range(2):
            r = lengths[k] / ref[k] if np.isfinite(ref[k]) else np.nan
            good = lo <= lengths[k] <= hi and (not np.isfinite(r) or abs(r - 1.0) <= self.tol)
            if good and learn[k]:
                self.hist[k].append(lengths[k])
            ok.append(good)
            ratio.append(r)
        return ok[0], ok[1], {"len": lengths, "ref": ref, "ratio": ratio, "ok": ok}


class PointKalman:
    """Kalman vận tốc không đổi cho N điểm 3D (mỗi toạ độ độc lập, cùng hiệp phương sai). Thay EMA điểm mốc.

    q: độ lệch chuẩn gia tốc (m/s^2) - lớn = bám nhanh, ít mượt. r: nhiễu đo (m) khi độ tin cậy = 1; độ tin cậy thấp
    -> nhiễu đo lớn hơn (r / conf) -> tin dự đoán hơn. Trả vị trí đã lọc (không ngoại suy tới tương lai). Mất điểm
    lâu hơn max_gap_s -> khởi tạo lại (không đi tiếp theo vận tốc cũ)."""

    def __init__(self, q=6.0, r=0.015, max_gap_s=0.3):
        self.q, self.r, self.max_gap = float(q), float(r), float(max_gap_s)
        self.reset()

    def reset(self):
        self.x = self.v = None
        self.P = None          # [P_pp, P_pv, P_vv]
        self.t = None

    def __call__(self, z, t, conf=1.0):
        z = np.asarray(z, float)
        if self.x is None or z.shape != self.x.shape or t - self.t > self.max_gap or t <= self.t:
            self.x, self.v, self.t = z.copy(), np.zeros_like(z), t
            self.P = [self.r ** 2, 0.0, 1.0]
            return self.x
        dt = t - self.t
        self.t = t
        pp, pv, vv = self.P
        q2 = self.q ** 2
        # Dự đoán: x += v dt; P = F P F^T + Q (gia tốc trắng)
        self.x = self.x + self.v * dt
        pp, pv, vv = (pp + 2 * dt * pv + dt * dt * vv + q2 * dt ** 4 / 4,
                      pv + dt * vv + q2 * dt ** 3 / 2,
                      vv + q2 * dt * dt)
        R = (self.r / max(float(conf), 0.05)) ** 2
        k_p, k_v = pp / (pp + R), pv / (pp + R)
        y = z - self.x
        self.x = self.x + k_p * y
        self.v = self.v + k_v * y
        self.P = [(1 - k_p) * pp, (1 - k_p) * pv, vv - k_v * pv]
        return self.x
