"""Lọc nhiễu theo từng khớp.

- One Euro filter (Casiez, Roussel, Vogel, CHI 2012): êm khi đứng yên, nhanh khi di chuyển.
- Vùng chết có trễ (hysteresis): rung nhỏ không tới robot.
- Bỏ bước nhảy: đổi quá `jump_deg` trong một khung coi là lỗi tracking, giữ tới khi lặp lại đủ lâu.
- Độ tin cậy riêng từng khớp: khớp nào tin cậy thấp thì chỉ khớp đó đứng yên.
Ý tưởng từ Im-ma/Teleoperation-Arm (filters.mjs); code tự viết.
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
