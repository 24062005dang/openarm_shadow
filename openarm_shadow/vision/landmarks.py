"""Chỉ số landmark MediaPipe Pose và Hand dùng trong repo."""
from __future__ import annotations


# chỉ số MediaPipe Pose
NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = 0, 11, 12, 13, 14, 15, 16, 23, 24


ARM_IDX = {"left": (L_SH, L_EL, L_WR), "right": (R_SH, R_EL, R_WR)}


# chỉ số MediaPipe Hand
H_WRIST, H_THUMB_TIP, H_INDEX_MCP, H_INDEX_TIP, H_MIDDLE_MCP, H_PINKY_MCP = 0, 4, 5, 8, 9, 17


FINGER_CHAINS = ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
