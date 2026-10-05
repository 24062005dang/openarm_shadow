"""Tự engage sau khi người điều khiển giữ tư thế READY liên tục đủ lâu (mỗi lần chạy chương trình tối đa một lần).

READY = mọi tay robot đã hiệu chuẩn hướng bàn tay VÀ khung hiện tại tay người đang đúng tư thế hiệu chuẩn (tay buông,
xoè, lòng bàn tay nhìn camera). Mất READY thì đếm lại. Người điều khiển bấm SPACE (engage/nhả thủ công) thì thôi tự
engage. Thời gian giữ: calibration.hand_auto.auto_engage_sim_s (mô phỏng) / auto_engage_real_s (robot thật, null = tắt).
"""
from __future__ import annotations


class AutoEngage:
    def __init__(self, hold_s):
        self.hold_s = None if hold_s is None else float(hold_s)
        self.used = False             # đã tự engage hoặc người điều khiển đã bấm SPACE
        self.ready_since = None
        self.countdown = None         # giây còn lại tới khi tự engage (None = không đếm)

    @classmethod
    def from_config(cls, cfg, robot_kind):
        hc = cfg.get("calibration", {}).get("hand_auto", {})
        from ..robot import REAL_KINDS
        return cls(hc.get("auto_engage_real_s" if robot_kind in REAL_KINDS else "auto_engage_sim_s", None))

    @property
    def enabled(self):
        return self.hold_s is not None

    def hint(self):
        """Phần đầu dòng hướng dẫn trên màn hình."""
        return f"GIU READY {self.hold_s:g}s: tu dong sync | " if self.enabled else ""

    def manual(self):
        """Người điều khiển bấm SPACE: từ giờ chỉ engage/nhả thủ công."""
        self.used, self.ready_since, self.countdown = True, None, None

    def update(self, ready, engaged, now):
        """Gọi mỗi khung. -> True nếu phải engage ngay bây giờ."""
        if not self.enabled or engaged or self.used:
            return False
        if not ready:
            self.ready_since = self.countdown = None
            return False
        self.ready_since = now if self.ready_since is None else self.ready_since
        self.countdown = max(0.0, self.hold_s - (now - self.ready_since))
        if self.countdown > 0.0:
            return False
        self.used, self.countdown = True, None
        return True
