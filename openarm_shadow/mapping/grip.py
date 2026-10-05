"""Độ mở kẹp từ bàn tay: r = khoảng cách đầu ngón cái - đầu ngón trỏ / chiều dài bàn tay (không phụ thuộc tay to
nhỏ, xa gần). Chụm = 0 (đóng), xoè = 1 (mở hẳn).

- Liên tục (mặc định): kẹp mở theo đúng độ hở hai ngón.
- Theo mức (levels, vd [0, 0.5, 1]): chỉ nhận vài mức; đổi mức cần vượt ngưỡng thêm `hysteresis` và giữ `dwell_s`
  (tay ở sát ranh giới không làm kẹp nhảy qua lại).
- Hiệu chuẩn theo người (phím g khi chạy): trong `calib_s` giây chụm hết cỡ rồi xoè hết cỡ vài lần; lấy r nhỏ
  nhất/lớn nhất (bỏ 5% hai đầu cho nhiễu) làm pinch/open, chừa `margin` để chụm/xoè thoải mái vẫn tới 0/1.
- Không thấy bàn tay: trả NaN (bộ lọc giữ kẹp ở giá trị cũ, không tự đóng/mở).
"""
from __future__ import annotations

import numpy as np


class GripMapper:
    def __init__(self, pinch_ratio=0.25, open_ratio=0.9, levels=None, hysteresis=0.05, dwell_s=0.15,
                 calib_s=4.0, margin=0.08):
        self.pinch, self.open = float(pinch_ratio), float(open_ratio)
        self.levels = None if not levels else sorted(float(v) for v in levels)
        self.hyst, self.dwell, self.calib_s, self.margin = float(hysteresis), float(dwell_s), float(calib_s), margin
        self.level_idx = None
        self._pending = None            # (chỉ số mức mới, thời điểm bắt đầu)
        self.r = np.nan                 # tỉ số ngón cái-trỏ khung vừa rồi (hiển thị)
        self.value = np.nan             # độ mở 0..1 khung vừa rồi
        self._calib_until, self._calib_r = None, []
        self.calib_result = None        # (pinch, open) hoặc chuỗi lỗi, sau khi hiệu chuẩn xong

    # -- hiệu chuẩn ----------------------------------------------------------------------------------
    def start_calibration(self, t):
        self._calib_until, self._calib_r, self.calib_result = t + self.calib_s, [], None

    def calib_remaining(self, t):
        return None if self._calib_until is None else max(0.0, self._calib_until - t)

    def _finish_calibration(self):
        self._calib_until = None
        r = np.asarray(self._calib_r, float)
        if len(r) < 10:
            self.calib_result = "không đủ khung thấy bàn tay"
            return
        lo, hi = np.percentile(r, 5), np.percentile(r, 95)
        if hi - lo < 0.25:
            self.calib_result = f"chụm/xoè chưa đủ rộng (r {lo:.2f}..{hi:.2f})"
            return
        span = hi - lo
        self.pinch, self.open = float(lo + self.margin * span), float(hi - self.margin * span)
        self.calib_result = (self.pinch, self.open)

    # -- ánh xạ --------------------------------------------------------------------------------------
    def continuous(self, r):
        return float(np.clip((r - self.pinch) / max(self.open - self.pinch, 1e-6), 0.0, 1.0))

    def _quantize(self, g, t):
        lv = self.levels
        if self.level_idx is None:
            self.level_idx = int(np.argmin([abs(g - v) for v in lv]))
            return lv[self.level_idx]
        k = self.level_idx
        target = k
        while target + 1 < len(lv) and g > 0.5 * (lv[target] + lv[target + 1]) + self.hyst:
            target += 1
        while target - 1 >= 0 and g < 0.5 * (lv[target - 1] + lv[target]) - self.hyst:
            target -= 1
        if target == k:
            self._pending = None
        elif self._pending is None or self._pending[0] != target:
            self._pending = (target, t)
        elif t - self._pending[1] >= self.dwell:
            self.level_idx, self._pending = target, None
        return lv[self.level_idx]

    def __call__(self, r, t):
        """r: tỉ số ngón cái-trỏ (None/NaN = không thấy bàn tay). Trả độ mở 0..1 hoặc NaN."""
        if self._calib_until is not None and t >= self._calib_until:
            self._finish_calibration()
        if r is None or not np.isfinite(r):
            self.r = self.value = np.nan
            self._pending = None
            return np.nan
        self.r = float(r)
        if self._calib_until is not None:
            self._calib_r.append(self.r)
        g = self.continuous(self.r)
        self.value = self._quantize(g, t) if self.levels else g
        return self.value
