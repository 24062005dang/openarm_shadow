"""Chỉ số điểm mốc MediaPipe và kiểu dữ liệu quan sát tay người (ArmObs, Frame) dùng chung mọi khâu."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.geometry import unit

# chỉ số MediaPipe Pose
NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = 0, 11, 12, 13, 14, 15, 16, 23, 24
ARM_IDX = {"left": (L_SH, L_EL, L_WR), "right": (R_SH, R_EL, R_WR)}
# chỉ số MediaPipe Hand
H_WRIST, H_THUMB_TIP, H_INDEX_MCP, H_INDEX_TIP, H_MIDDLE_MCP, H_PINKY_MCP = 0, 4, 5, 8, 9, 17
FINGER_CHAINS = ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))


@dataclass
class ArmObs:
    s: np.ndarray | None = None
    e: np.ndarray | None = None
    w: np.ndarray | None = None
    H: np.ndarray | None = None
    grip: float | None = None
    conf: dict = field(default_factory=lambda: {"upper": 0.0, "fore": 0.0, "hand": 0.0})
    hand_points_cam: np.ndarray | None = None  # (21,3), metric, cùng camera frame D455; NaN nếu thiếu
    hand_depth_valid: int = 0
    hand_depth_confidence: float = 0.0
    hand_depth_mode: str = "NONE"
    hand_R_cam: np.ndarray | None = None
    hand_center_cam: np.ndarray | None = None
    hand_axes_px: np.ndarray | None = None  # origin,x,y,z endpoints trong pixel D455 RGB
    hand_open_fingers: int = 0
    hand_orientation_mode: str = "NONE"

    @property
    def u(self):
        return None if self.s is None else unit(self.e - self.s)

    @property
    def l(self):
        return None if self.e is None else unit(self.w - self.e)


@dataclass
class Frame:
    arms: dict                     # "right"/"left" -> ArmObs
    pose_2d: np.ndarray | None     # (33, 3) x, y chuẩn hoá + visibility, để vẽ
    hands_2d: list                 # [(21, 2) array, side hoặc None]
    body_R: np.ndarray | None      # cột = trục khung thân trong toạ độ camera
    t: float = 0.0
    depth_used: dict = field(default_factory=dict)  # số landmark depth hợp lệ theo tay (0..3)
    hand_depth: dict = field(default_factory=dict)  # diagnostics 21 điểm bàn tay theo tay người
    body_origin: np.ndarray | None = None
    fusion: dict = field(default_factory=dict)      # chẩn đoán khi hợp nhất nhiều camera (fusion/multiview.py)
