"""Bộ theo dõi hướng bàn tay chống lật dấu khi nhiều camera."""
from __future__ import annotations

import numpy as np

from openarm_shadow.core.rotation import rotation_distance, slerp_rotation

FLIP_X = np.diag([1.0, -1.0, -1.0])        # quay 180° quanh trục ngón tay (x của khung bàn tay)


# ----------------------------------------------------------------------------------------------------------
# Hướng bàn tay: giữ liên tục dấu
# ----------------------------------------------------------------------------------------------------------
class HandOrientationTracker:
    """Làm mượt hướng bàn tay và chống lật úp/ngửa.

    - Khi mọi camera đều nhìn cạnh bàn tay (quality thấp), pháp tuyến có thể lật dấu: chọn trong hai giả thuyết
      R và R quay 180° quanh trục ngón cái nào gần hướng trước hơn.
    - Nhảy quá max_jump_deg trong 1 khung: giữ hướng cũ, chỉ nhận khi lặp lại confirm_frames khung liên tiếp.
    - Mất quan sát: giữ hướng cũ tối đa hold_frames khung rồi mới báo NONE (không tự nhận hướng lật).
    """

    def __init__(self, alpha=0.6, hold_frames=8, ambiguous_quality=0.35, max_jump_deg=100.0, confirm_frames=4,
                 alpha_max=0.85):
        # alpha: tỉ lệ nhận hướng mới khi nhìn KÉM (nhiều làm mượt); alpha_max: khi nhìn rõ (ít trễ, để cổ tay
        # không chậm hơn cánh tay; bộ lọc từng khớp phía sau vẫn làm mượt).
        self.alpha, self.alpha_max, self.hold, self.amb_q = alpha, max(alpha, alpha_max), hold_frames, ambiguous_quality
        self.max_jump, self.confirm = np.deg2rad(max_jump_deg), confirm_frames
        self.R, self.miss, self.jumps = None, 0, 0

    def update(self, R, quality):
        if R is None:
            self.miss += 1
            if self.R is not None and self.miss <= self.hold:
                return self.R, "HOLD"
            self.R = None
            return None, "NONE"
        self.miss = 0
        if self.R is None:
            self.R, self.jumps = R, 0
            return R, "FUSED"
        mode = "FUSED"
        d, R_alt = rotation_distance(self.R, R), R @ FLIP_X
        d_alt = rotation_distance(self.R, R_alt)
        if quality < self.amb_q and d_alt < d:
            R, d, mode = R_alt, d_alt, "SIGN-FIX"
        if d > self.max_jump:
            self.jumps += 1
            if self.jumps < self.confirm:
                return self.R, "HOLD"
        self.jumps = 0
        k = float(np.clip((quality - self.amb_q) / max(1.0 - self.amb_q, 1e-6), 0.0, 1.0))
        self.R = slerp_rotation(self.R, R, self.alpha + (self.alpha_max - self.alpha) * k)
        return self.R, mode
