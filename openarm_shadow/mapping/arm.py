"""Một tay OpenArm v1.0 (trái hoặc phải): động học, bộ giải retarget, bộ lọc, kẹp, trạng thái hiệu chuẩn, giới hạn khớp.

Trước đây ShadowPipeline giữ ~20 dict riêng theo `side`; nay mỗi tay là một đối tượng `Arm`, pipeline chỉ lặp qua
`pipe.arms`. Tay trái và tay phải dùng CHUNG một lớp, khác nhau ở dữ liệu (URDF từng tay) và ở `MIRROR_SIGNS`.

Quan hệ trái / phải (đọc từ URDF `data/openarm_v10_arms.json`, bảng giới hạn trong `ROS inference.md` và MJCF
`openarm_mujoco/v1/openarm_bimanual.xml`; tests/test_arm.py kiểm tra cả ba nguồn khớp nhau):

    khớp  giới hạn trái so với phải      trục khớp (khung khớp)   đảo dấu khi phản chiếu tư thế
    J1    đảo:  trái = -phải             giống nhau               -1   (+q: phải ra trước, trái ra sau)
    J2    đảo:  trái = -phải             giống nhau               -1
    J3    đối xứng (±90°)                giống nhau               -1
    J4    giống nhau (0..140°)           giống nhau               +1   (gập khuỷu, cùng chiều hai tay)
    J5    đối xứng (±90°)                giống nhau               -1
    J6    đối xứng (±45°)                giống nhau               -1
    J7    đối xứng (±90°)                trái = -phải             -1

Chỉ J1, J2 có GIỚI HẠN ngược nhau, nhưng phép phản chiếu TƯ THẾ (tay trái làm ảnh gương của tay phải) đảo dấu mọi
khớp trừ J4: J3, J5, J6 có giới hạn đối xứng nên nhìn bảng giới hạn không thấy, và trục J7 của tay trái ngược tay phải.
Chỉ đảo J1, J2 thì tay trái xoay cẳng tay / cổ tay sai chiều (sai tới ~157° khi cả 7 khớp cử động).
`mirror_q` chính xác cho HƯỚNG các link; vị trí khớp lệch tới ~6 cm vì URDF không lật các độ lệch cơ khí giữa link
(tay trái là bản xoay của tay phải, không phải bản lật gương) - retarget chỉ dùng hướng nên không ảnh hưởng.
"""
from __future__ import annotations

import numpy as np

from openarm_shadow.filtering.filters import EMA, ArmShape, JointFilter, PointKalman
from openarm_shadow.mapping.grip import GripMapper
from openarm_shadow.core.kinematics import ArmKinematics
from openarm_shadow.mapping.retarget import ArmRetargeter, default_hand_neutral, mirror_rotation, palm_forward_hand

SIDES = ("right", "left")

# q_trái = MIRROR_SIGNS * q_phải (và ngược lại: phép đảo dấu là nghịch đảo của chính nó). Hướng link 7 trùng khít
# ảnh gương (qua mặt phẳng dọc giữa thân, y -> -y) của tay kia, sai lệch 0 (tests/test_arm.py).
MIRROR_SIGNS = np.array([-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0])


def other_side(side):
    return "left" if side == "right" else "right"


def mirror_limits_deg(limits):
    """Giới hạn [[lo, hi]] x 7 (độ, góc URDF) của một tay -> giới hạn tương ứng của tay kia khi nó làm ảnh gương:
    khớp đảo dấu (MIRROR_SIGNS = -1) thì [lo, hi] -> [-hi, -lo]. Vd phải J2 [-9, 90] -> trái [-90, 9]."""
    return [[float(min(s * lo, s * hi)), float(max(s * lo, s * hi))] for (lo, hi), s in zip(limits, MIRROR_SIGNS)]


def mirror_q(q):
    """Góc khớp của tay kia khi nó làm ảnh gương của tư thế q. q: 7 góc, hoặc 8 phần tử (kẹp 0..1 giữ nguyên);
    NaN (giữ khớp) giữ nguyên NaN."""
    q = np.asarray(q, float)
    out = q.copy()
    out[:7] = MIRROR_SIGNS * q[:7]
    return out


class Arm:
    """Một tay robot. side: "right" | "left". human_side: tay người điều khiển tay này. mirrored: True khi quan sát
    của tay người bị phản chiếu (chế độ mapping.mode = mirror)."""

    def __init__(self, side: str, cfg: dict, human_side: str | None = None, mirrored: bool = False):
        assert side in SIDES
        self.side = side
        self.human_side = human_side or side
        self.mirrored = bool(mirrored)
        self.kin = ArmKinematics(side)
        rc = cfg["retarget"]
        self.rt = ArmRetargeter(self.kin, rc["elbow_straight_deg"],
                                 shoulder_singular_deg=rc.get("shoulder_singular_deg", (5.0, 20.0)))
        # hướng trung tính mặc định theo đúng bàn tay người điều khiển tay này
        Hn = default_hand_neutral(self.human_side)
        self.rt.set_hand_neutral(mirror_rotation(Hn) if self.mirrored else Hn)
        # Góc J5-J7 robot ứng với tư thế HIỆU CHUẨN (tay xuôi, lòng bàn tay nhìn camera) theo ánh xạ hình học trên:
        # tay xuôi + ngón cái ra trước <-> J5-J7 = 0, khi đó hai ngón kẹp đóng / mở theo PHÁP TUYẾN lòng bàn tay
        # (mặt phẳng hai ngón kẹp vuông góc lòng bàn tay, như ngón cái - ngón trỏ khi gắp). Lòng bàn tay ra trước là
        # ngửa cổ tay 90° so với tư thế đó -> J5 = ±90°. Hiệu chuẩn gán tư thế đo được với góc này (không phải
        # J5-J7 = 0 như trước: kẹp bị lệch 90°, đóng / mở trong mặt phẳng bàn tay). Ngón cái ra trước nằm giữa tầm
        # xoay cổ tay người nên dùng được hết J5 ±90°.
        Hc = palm_forward_hand(self.human_side)
        down = np.array([0.0, 0.0, -1.0])
        q_cal, _ = self.rt.solve(down, down, mirror_rotation(Hc) if self.mirrored else Hc, np.zeros(7))
        self.calib_wrist = q_cal[4:7].copy()

        fc = cfg["filter"]
        self.filt = JointFilter(8, fc["min_cutoff"], fc["beta"], fc["deadband_deg"], fc["jump_deg"],
                                fc["jump_hold_s"], fc["min_conf"], angular=[True] * 7 + [False],
                                jump_confirm_conf=fc.get("jump_confirm_conf", 0.0))
        self.lm_ema = EMA(fc["landmark_ema_alpha"])
        # Kalman điểm 3D thay EMA (tuỳ chọn), lọc khung xương cánh tay (tuỳ chọn; cần điểm thang mét nhất quán)
        kc = dict(fc.get("landmark_kalman") or {})
        self.lm_kf = (PointKalman(kc.get("q", 6.0), kc.get("r", 0.015), kc.get("max_gap_s", 0.3),
                                  kc.get("gate_sigma", 0.0), kc.get("gate_min_m", 0.05), kc.get("confirm_frames", 3))
                      if kc.get("enabled", False) else None)
        ac = dict(fc.get("arm_shape") or {})
        self.arm_shape = (ArmShape(ac.get("tol", 0.25), ac.get("samples", 90), ac.get("min_samples", 15),
                                   ac.get("reset_s", 2.0), ac.get("len_range_m", (0.12, 0.5)))
                          if ac.get("enabled", False) else None)
        g = cfg["grip"]
        self.grip = GripMapper(g["pinch_ratio"], g["open_ratio"], g.get("levels"), g.get("level_hysteresis", 0.05),
                               g.get("level_dwell_s", 0.15), g.get("calib_s", 4.0),
                               release_s=g.get("release_confirm_s", 0.3),
                               move_speed_mps=g.get("release_move_speed_mps", 0.35),
                               release_max_s=g.get("release_max_s", 1.5),
                               release_reacquire_s=g.get("release_reacquire_s", 1.0))

        # trạng thái theo khung
        self.q_prev = np.zeros(7)
        self.last_info = None
        self.held = np.ones(8, bool)             # cờ giữ của bộ lọc từng khớp (SafetyGate)
        self.conf = np.zeros(8)                  # độ tin cậy từng khớp khung vừa rồi (ghi --record)
        self.kf_rejected = np.zeros(3, bool)     # vai/khuỷu/cổ tay bị Kalman loại khung vừa rồi
        self.arm_shape_info = None
        # hiệu chuẩn hướng bàn tay
        self.hand_calibrated = False
        self.calib_ready_now = False
        self.calib_progress = 0.0
        self.calib_hint = "dua tay vao khung"
        self._calib_samples = []
        self._calib_start = None
        self._calib_prev = None

    # -- giới hạn / chuyển đổi ---------------------------------------------------------------------------------------
    @property
    def lower(self):
        return self.kin.lower

    @property
    def upper(self):
        return self.kin.upper

    @property
    def partner(self):
        return other_side(self.side)

    def reset_auto_calib(self, keep_progress=False):
        self._calib_samples = []
        self._calib_start = None
        self._calib_prev = None
        if not keep_progress:
            self.calib_progress = 0.0

    def __repr__(self):
        return f"Arm({self.side}, human={self.human_side}{', mirrored' if self.mirrored else ''})"


class _SideView:
    """Truy cập kiểu cũ `pipe.<tên>[side]` trỏ thẳng vào thuộc tính của Arm (đọc, gán, get, items...), để mã và test
    dùng dict theo tay (pipe.held["right"], pipe.rt["left"]...) chạy tiếp mà không giữ bản sao trạng thái."""

    def __init__(self, arms: dict, name: str):
        self._arms, self._name = arms, name

    def __getitem__(self, side):
        return getattr(self._arms[side], self._name)

    def __setitem__(self, side, value):
        setattr(self._arms[side], self._name, value)

    def __contains__(self, side):
        return side in self._arms and getattr(self._arms[side], self._name) is not None

    def __iter__(self):
        return iter(self._arms)

    def __len__(self):
        return len(self._arms)

    def keys(self):
        return self._arms.keys()

    def values(self):
        return [getattr(a, self._name) for a in self._arms.values()]

    def items(self):
        return [(s, getattr(a, self._name)) for s, a in self._arms.items()]

    def get(self, side, default=None):
        a = self._arms.get(side)
        v = getattr(a, self._name) if a is not None else None
        return default if v is None else v
