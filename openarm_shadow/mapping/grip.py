"""Độ mở kẹp từ bàn tay: r = khoảng cách đầu ngón cái - đầu ngón trỏ / chiều dài bàn tay (không phụ thuộc tay to
nhỏ, xa gần). Chụm = 0 (đóng), xoè = 1 (mở hẳn).

- Liên tục (mặc định): kẹp mở theo đúng độ hở hai ngón.
- Theo mức (levels, vd [0, 0.5, 1]): chỉ nhận vài mức; đổi mức cần vượt ngưỡng thêm `hysteresis` và giữ `dwell_s`
  (tay ở sát ranh giới không làm kẹp nhảy qua lại).
- Hiệu chuẩn theo người (phím g khi chạy): trong `calib_s` giây chụm hết cỡ rồi xoè hết cỡ vài lần; lấy r nhỏ
  nhất/lớn nhất (bỏ 5% hai đầu cho nhiễu) làm pinch/open, chừa `margin` để chụm/xoè thoải mái vẫn tới 0/1.
- Không thấy bàn tay: trả NaN (bộ lọc giữ kẹp ở giá trị cũ, không tự đóng/mở).
- Chốt khi đang kẹp: đóng nhận ngay, nhưng MỞ từ trạng thái đang kẹp (<= latch_below) thêm > release_delta phải giữ
  liên tục release_s mới nhận (khung mất tay ở giữa không xoá đồng hồ). Tay di chuyển nhanh làm ảnh nhoè, MediaPipe
  đặt sai đầu ngón -> r nhảy "mở hẳn" 1-2 khung với độ tin cậy cao (run5 05/10: 0,1-0,25 s) -> trước đây kẹp nhả vật.
- Theo chuyển động (wrist = vị trí cổ tay, mét): bàn tay còn đi nhanh hơn move_speed_mps trong release_s vừa qua thì
  chưa nhả (đang mang / chuyền vật thì ảnh nhoè lâu, đo nhầm "mở" có lúc > 0,3 s; người thường dừng tay rồi mới thả).
  Mở liên tục quá release_max_s thì nhả dù tay còn đi (cố ý thả khi đang di chuyển không bị kẹt mãi).
- Nhiều camera (r_views): đang kẹp thì xét nhả theo camera thấy CHỤM NHẤT (một camera đặt sai đầu ngón không đủ để nhả).
- Mất bàn tay khi đang kẹp: bắt lại xong phải thấy mở liên tục release_reacquire_s mới nhả. MediaPipe bắt lại một bàn
  tay đang chụm hay đoán thành bàn tay xoè, sai ổn định nhiều giây (run6 06/10), chốt release_s không đủ.
"""
from __future__ import annotations

import numpy as np


class GripMapper:
    def __init__(self, pinch_ratio=0.25, open_ratio=0.9, levels=None, hysteresis=0.05, dwell_s=0.15,
                 calib_s=4.0, margin=0.08, release_s=0.0, latch_below=0.3, release_delta=0.25,
                 move_speed_mps=0.35, release_max_s=1.5, release_reacquire_s=1.0):
        self.pinch, self.open = float(pinch_ratio), float(open_ratio)
        self.levels = None if not levels else sorted(float(v) for v in levels)
        self.hyst, self.dwell, self.calib_s, self.margin = float(hysteresis), float(dwell_s), float(calib_s), margin
        self.level_idx = None
        self._pending = None            # (chỉ số mức mới, thời điểm bắt đầu)
        self.r = np.nan                 # tỉ số ngón cái-trỏ khung vừa rồi (hiển thị)
        self.value = np.nan             # độ mở 0..1 khung vừa rồi
        self._calib_until, self._calib_r = None, []
        self.release_s, self.latch_below, self.release_delta = float(release_s), float(latch_below), float(release_delta)
        self._accepted = np.nan        # độ mở đã nhận gần nhất (chốt kẹp)
        self._release_since = None     # đang chờ xác nhận mở từ lúc nào
        self.releasing = False         # hiển thị: đang chờ xác nhận nhả
        self.move_speed, self.release_max_s = float(move_speed_mps), float(release_max_s)
        self.speed = 0.0               # tốc độ cổ tay (m/s, làm mượt)
        self._w_prev = None            # (vị trí cổ tay, t)
        self._fast_t = -np.inf         # lần cuối bàn tay đi nhanh hơn move_speed
        self.release_reacquire_s = float(release_reacquire_s)
        self._lost_latched = False     # đã mất bàn tay trong lúc đang kẹp (chưa nhả lại / chưa đo chụm lại)
        self._lost_since = None        # mất 1 khung lẻ (< 0,15 s) không tính là mất tay
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

    def _update_motion(self, wrist, t):
        if wrist is None or not np.all(np.isfinite(wrist)):
            return
        w = np.asarray(wrist, float)
        if self._w_prev is not None and 0.0 < t - self._w_prev[1] < 0.5:
            v = float(np.linalg.norm(w - self._w_prev[0]) / (t - self._w_prev[1]))
            self.speed = 0.5 * v + 0.5 * self.speed       # nhiễu vị trí fusion 1-2 cm/khung: làm mượt
            if self.speed > self.move_speed:
                self._fast_t = t
        self._w_prev = (w.copy(), t)

    def _is_latched(self):
        a = self._accepted
        return self.release_s > 0 and self._calib_until is None and np.isfinite(a) and a <= self.latch_below

    def __call__(self, r, t, wrist=None, r_views=None):
        """r: tỉ số ngón cái-trỏ (None/NaN = không thấy bàn tay). wrist: vị trí cổ tay (m) để biết tay đang di
        chuyển (tuỳ chọn). r_views: tỉ số từng camera (tuỳ chọn). Trả độ mở 0..1 hoặc NaN."""
        self._update_motion(wrist, t)
        if r is None or not np.isfinite(r):
            if self._is_latched():
                self._lost_since = t if self._lost_since is None else self._lost_since
                if t - self._lost_since >= 0.15:
                    self._lost_latched = True
        else:
            self._lost_since = None
        if r is not None and np.isfinite(r) and self._is_latched() and r_views:
            rv = [x for x in r_views if x is not None and np.isfinite(x)]
            if rv:
                r = min(float(r), min(rv))         # đang kẹp: xét nhả theo camera thấy chụm nhất
        if self._calib_until is not None and t >= self._calib_until:
            self._finish_calibration()
        if r is None or not np.isfinite(r):
            self.r = self.value = np.nan
            self._pending = None
            return np.nan               # chờ nhả (nếu có) vẫn tiếp tục: mất tay giữa chừng không làm nhả
        self.r = float(r)
        if self._calib_until is not None:
            self._calib_r.append(self.r)
        g = self.continuous(self.r)
        v = self._quantize(g, t) if self.levels else g
        self.value = self._latch(v, t)
        return self.value

    def _latch(self, v, t):
        a = self._accepted
        latched = self._is_latched()
        if latched and v > a + self.release_delta:
            if self._release_since is None:
                self._release_since = t
            held_open = t - self._release_since
            moving = t - self._fast_t < self.release_s
            need = max(self.release_s, self.release_reacquire_s if self._lost_latched else 0.0)
            if held_open < need or (moving and held_open < self.release_max_s):
                self.releasing = True
                return a                # chưa đủ lâu: giữ kẹp
        self._release_since, self.releasing = None, False
        self._lost_latched = False                 # thấy lại chụm / đã nhả thật / không kẹp: hết nghi ngờ
        self._accepted = v
        return v
