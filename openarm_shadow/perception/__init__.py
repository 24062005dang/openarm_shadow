"""Nhận diện người trên MỘT camera: MediaPipe Pose + Hand (landmarker), khoá người điều khiển (operator), gán bàn
tay theo cổ tay (hand_assign), depth RealSense tại điểm mốc (depth), hình học bàn tay (hand_geometry), khung thân
(body). Kiểu dữ liệu đầu ra dùng chung mọi khâu: types.ArmObs, types.Frame."""
from .body import BodyRef, body_frame
from .depth import (deproject_pixel, fit_metric_depth, fit_palm_plane, fuse_hand_landmarks, project_point,
                    sample_depth, sample_depth_with_confidence)
from .hand_assign import assign_hands_to_wrists
from .hand_geometry import open_finger_count, palm_frame_from_depth
from .landmarker import Perception
from .operator import select_operator, shoulders
from .types import (ARM_IDX, FINGER_CHAINS, H_INDEX_MCP, H_INDEX_TIP, H_MIDDLE_MCP, H_PINKY_MCP, H_THUMB_TIP,
                    H_WRIST, L_EL, L_HIP, L_SH, L_WR, NOSE, R_EL, R_HIP, R_SH, R_WR, ArmObs, Frame)
