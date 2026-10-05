"""Lọc nhiễu theo từng khớp.

- One Euro filter (Casiez, Roussel, Vogel, CHI 2012): êm khi đứng yên, nhanh khi di chuyển.
- Vùng chết có trễ (hysteresis): rung nhỏ không tới robot.
- Bỏ bước nhảy: đổi quá `jump_deg` trong một khung coi là lỗi tracking, giữ tới khi lặp lại đủ lâu.
- Độ tin cậy riêng từng khớp: khớp nào tin cậy thấp thì chỉ khớp đó đứng yên.
- Lọc khung xương cánh tay (ArmShape): đoạn tay dài/ngắn bất thường so với độ dài đã học -> không tin đoạn đó.
- Kalman điểm 3D (PointKalman, tuỳ chọn): vai/khuỷu/cổ tay, nhiễu đo theo độ tin cậy.
Các ý tưởng xử lý lấy cảm hứng từ repo Im-ma/Teleoperation-Arm (filters.mjs) và bài
Hand Shadowing (arXiv 2603.11383, EMA hai tầng); code ở đây tự viết.
"""
from __future__ import annotations

import math

import numpy as np


class OneEuro:
    def __init__(self, min_cutoff=1.0, beta=0.02, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.t is None:
            self.x, self.dx, self.t = x, 0.0, t
            return x
        dt = max(t - self.t, 1e-3)
        self.t = t
        dx = (x - self.x) / dt
        self.dx += self._alpha(self.d_cutoff, dt) * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * abs(self.dx)
        self.x += self._alpha(cutoff, dt) * (x - self.x)
        return self.x


class JointFilter:
    """Lọc một vector góc (rad). Cấu hình theo từng phần tử."""

    def __init__(self, n, min_cutoff, beta, deadband_deg, jump_deg=35.0, jump_hold_s=0.2,
                 min_conf=0.6, angular=None, jump_confirm_conf=0.0):
        """angular[i] = False: phần tử i không phải góc (vd độ mở kẹp 0..1): deadband/jump dùng nguyên đơn vị,
        không đổi độ -> rad. Kẹp chỉ bỏ bước nhảy khi jump_deg của nó <= 1 (mặc định 35 -> tắt).

        Bước nhảy > jump_deg chỉ được nhận khi trong suốt jump_hold_s: giá trị mới ỔN ĐỊNH (lệch nhau < jump_deg/2,
        không phải nhiễu nhảy qua lại) và độ tin cậy >= jump_confirm_conf (không nhận bước nhảy từ điểm hợp nhất
        kém, vd chỉ 1 camera + depth). Không thoả: đếm lại từ đầu, khớp vẫn giữ."""
        as_list = lambda v: list(v) if np.ndim(v) else [v] * n
        self.n = n
        ang = np.ones(n, bool) if angular is None else np.asarray(angular, bool)
        self.f = [OneEuro(mc, b) for mc, b in zip(as_list(min_cutoff), as_list(beta))]
        self.dead = np.where(ang, np.deg2rad(as_list(deadband_deg)), np.asarray(as_list(deadband_deg), float))
        self.jump = np.where(ang, np.deg2rad(as_list(jump_deg)), np.asarray(as_list(jump_deg), float))
        self.jump_hold_s = jump_hold_s
        self.jump_conf = np.asarray(as_list(jump_confirm_conf), float)
        self.min_conf = min_conf
        self.reset()

    def reset(self):
        for f in self.f:
            f.reset()
        self.out = [None] * self.n
        self.raw = [None] * self.n      # (giá trị thô cuối, thời điểm)
        self.jump_since = [None] * self.n
        self.jump_val = [None] * self.n       # giá trị lúc bắt đầu bước nhảy đang chờ xác nhận
        self.held = np.ones(self.n, bool)

    def __call__(self, x, conf, t):
        """x: mảng n góc (có thể NaN), conf: mảng n độ tin cậy [0..1]. Trả về (out, held)."""
        for i in range(self.n):
            xi = x[i]
            if not np.isfinite(xi) or conf[i] < self.min_conf:
                self.held[i] = True
                self.jump_since[i] = None
                continue
            if self.raw[i] is not None and abs(xi - self.raw[i]) > self.jump[i]:
                if (self.jump_since[i] is None or conf[i] < self.jump_conf[i]
                        or abs(xi - self.jump_val[i]) > 0.5 * self.jump[i]):
                    self.jump_since[i], self.jump_val[i] = t, xi     # (bắt đầu) đếm lại
                if t - self.jump_since[i] < self.jump_hold_s or conf[i] < self.jump_conf[i]:
                    self.held[i] = True
                    continue          # chờ xem bước nhảy có lặp lại không
                self.f[i].reset()     # nhảy thật (lặp lại đủ lâu): bắt đầu lại bộ lọc
            self.jump_since[i] = None
            self.raw[i] = xi
            y = self.f[i](xi, t)
            o = self.out[i]
            if o is None or abs(y - o) > self.dead[i]:
                self.out[i] = y if o is None else y - math.copysign(self.dead[i], y - o)
            self.held[i] = False
        out = np.array([np.nan if o is None else o for o in self.out])
        return out, self.held.copy()


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
    lâu hơn max_gap_s -> khởi tạo lại (không đi tiếp theo vận tốc cũ).

    conf: một số cho mọi điểm, hoặc mảng N phần tử (độ tin cậy riêng từng điểm, vd vai/khuỷu/cổ tay): điểm nhìn kém
    tin dự đoán, điểm nhìn rõ vẫn bám đo. Hiệp phương sai vì thế tính riêng từng điểm.

    Loại điểm theo sai số dự đoán (innovation gating, gate_sigma > 0): điểm đo cách vị trí dự đoán quá gate_sigma độ
    lệch chuẩn (và quá gate_min_m) coi là sai (tay khác che, MediaPipe đặt nhầm chỗ mà vẫn báo "thấy rõ") -> bỏ đo,
    dùng dự đoán (vận tốc giảm một nửa để không trôi). Điểm bị loại liên tục confirm_frames khung thì chấp nhận (chuyển
    động nhanh thật, không kẹt ở dự đoán). self.rejected: điểm nào bị loại ở lần gọi vừa rồi."""

    def __init__(self, q=6.0, r=0.015, max_gap_s=0.3, gate_sigma=0.0, gate_min_m=0.05, confirm_frames=3):
        self.q, self.r, self.max_gap = float(q), float(r), float(max_gap_s)
        self.gate_sigma, self.gate_min = float(gate_sigma or 0.0), float(gate_min_m)
        self.confirm = int(confirm_frames)
        self.reset()

    def reset(self):
        self.x = self.v = None
        self.P = None          # [P_pp, P_pv, P_vv], mỗi phần tử dạng (N, 1): riêng từng điểm, chung 3 toạ độ
        self.t = None
        self.rejected = None
        self._out_n = None

    def _conf(self, conf, z):
        """Độ tin cậy -> mảng (N, 1) phát sóng được với z (N, 3)."""
        c = np.broadcast_to(np.asarray(conf, float), z.shape[:-1])
        return np.maximum(c, 0.05)[..., None]

    def __call__(self, z, t, conf=1.0):
        z = np.asarray(z, float)
        if self.x is None or z.shape != self.x.shape or t - self.t > self.max_gap or t <= self.t:
            self.x, self.v, self.t = z.copy(), np.zeros_like(z), t
            one = np.ones(z.shape[:-1] + (1,))
            self.P = [self.r ** 2 * one, 0.0 * one, one]
            self.rejected = np.zeros(z.shape[:-1], bool)
            self._out_n = np.zeros(z.shape[:-1], int)
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
        R = (self.r / self._conf(conf, z)) ** 2
        k_p, k_v = pp / (pp + R), pv / (pp + R)
        y = z - self.x
        out = np.zeros(z.shape[:-1], bool)
        if self.gate_sigma > 0:
            dist = np.linalg.norm(y, axis=-1)
            out = dist > np.maximum(self.gate_sigma * np.sqrt((pp + R)[..., 0]), self.gate_min)
            self._out_n = np.where(out, self._out_n + 1, 0)
            accept = out & (self._out_n >= self.confirm)
            if np.any(accept):                   # lệch liên tục: chuyển động thật -> nhận đo, khởi tạo lại điểm đó
                a = accept[..., None]
                self.x, self.v = np.where(a, z, self.x), np.where(a, 0.0, self.v)
                pp, pv, vv = np.where(a, self.r ** 2, pp), np.where(a, 0.0, pv), np.where(a, 1.0, vv)
                self._out_n = np.where(accept, 0, self._out_n)
                y = np.where(a, 0.0, y)
                out = out & ~accept
            o = out[..., None]
            k_p, k_v = np.where(o, 0.0, k_p), np.where(o, 0.0, k_v)
            self.v = np.where(o, 0.5 * self.v, self.v)
        self.rejected = out
        self.x = self.x + k_p * y
        self.v = self.v + k_v * y
        self.P = [(1 - k_p) * pp, (1 - k_p) * pv, vv - k_v * pv]
        return self.x
