"""MediaPipe Pose + Hand (Tasks API, chạy CPU) -> quan sát tay người trong khung thân.

Đầu ra cho mỗi tay người ("right" / "left" = tay phải/trái THẬT của người):
    s, e, w : vai, khuỷu, cổ tay (m) trong khung thân (x trước, y trái, z lên, gốc giữa hai vai)
    H       : 3x3 hướng bàn tay trong khung thân (cột x = hướng ngón, y = út→trỏ, z = x×y) hoặc None
    grip    : độ mở kẹp 0..1 (khoảng cách đầu ngón cái–trỏ / chiều dài bàn tay) hoặc None
    conf    : độ tin cậy cho 3 nhóm khớp: upper (J1–J2), fore (J3–J4), hand (J5–J7 + kẹp)

Lưu ý: chạy MediaPipe trên ảnh GỐC (không lật gương). Lật gương chỉ để hiển thị.
Giả định: điểm 3D "world" của Pose và Hand đều có hướng trục theo camera (gốc khác nhau),
nên hướng bàn tay có thể đổi sang khung thân bằng cùng ma trận quay.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .geometry import make_frame, unit

# chỉ số MediaPipe Pose
NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = 0, 11, 12, 13, 14, 15, 16, 23, 24
ARM_IDX = {"left": (L_SH, L_EL, L_WR), "right": (R_SH, R_EL, R_WR)}
# chỉ số MediaPipe Hand
H_WRIST, H_THUMB_TIP, H_INDEX_MCP, H_INDEX_TIP, H_MIDDLE_MCP, H_PINKY_MCP = 0, 4, 5, 8, 9, 17


@dataclass
class ArmObs:
    s: np.ndarray | None = None
    e: np.ndarray | None = None
    w: np.ndarray | None = None
    H: np.ndarray | None = None
    grip: float | None = None
    conf: dict = field(default_factory=lambda: {"upper": 0.0, "fore": 0.0, "hand": 0.0})

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


def hand_frame(hw):
    """hw: (21, 3) điểm world của bàn tay. Trả về 3x3 [x ngón, y út→trỏ, z]."""
    x = unit(hw[H_MIDDLE_MCP] - hw[H_WRIST])
    across = hw[H_INDEX_MCP] - hw[H_PINKY_MCP]
    y = unit(across - (across @ x) * x)
    return np.column_stack([x, y, np.cross(x, y)])


def body_frame(W, vis, min_hip_vis=0.5):
    """Khung thân từ điểm world của Pose. Thiếu hông (ngồi, bị bàn che) thì dùng 'lên' của camera."""
    if min(vis[L_HIP], vis[R_HIP]) >= min_hip_vis:
        bottom = 0.5 * (W[L_HIP] + W[R_HIP])
    else:
        bottom = 0.5 * (W[L_SH] + W[R_SH]) + np.array([0.0, 0.5, 0.0])   # y của MediaPipe hướng xuống
    return make_frame(W[L_SH], W[R_SH], bottom)


class Perception:
    def __init__(self, pose_model, hand_model, num_hands=2, min_conf=0.5):
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision

        self.mp = mp
        RM = vision.RunningMode.VIDEO
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=str(pose_model)), running_mode=RM,
            num_poses=1, min_pose_detection_confidence=min_conf,
            min_pose_presence_confidence=min_conf, min_tracking_confidence=min_conf))
        self.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=str(hand_model)), running_mode=RM,
            num_hands=num_hands, min_hand_detection_confidence=min_conf,
            min_hand_presence_confidence=min_conf, min_tracking_confidence=min_conf))
        self._last_ts = -1

    def close(self):
        self.pose.close()
        self.hands.close()

    def process(self, bgr, t=None) -> Frame:
        """bgr: ảnh OpenCV gốc (không lật). t: thời điểm (s), mặc định time.monotonic()."""
        t = time.monotonic() if t is None else t
        ts = max(int(t * 1000), self._last_ts + 1)
        self._last_ts = ts
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        pres = self.pose.detect_for_video(img, ts)
        hres = self.hands.detect_for_video(img, ts)
        h, w = bgr.shape[:2]

        arms = {"right": ArmObs(), "left": ArmObs()}
        if not pres.pose_landmarks:
            return Frame(arms, None, [], None, t)
        P = pres.pose_landmarks[0]
        pose_2d = np.array([[p.x, p.y, p.visibility or 0.0] for p in P])
        vis = pose_2d[:, 2]
        W = np.array([[p.x, p.y, p.z] for p in pres.pose_world_landmarks[0]])
        R_body, origin = body_frame(W, vis)
        to_body = lambda v: R_body.T @ (v - origin)

        # ghép bàn tay với cổ tay gần nhất trong ảnh (không tin nhãn handedness)
        sh_px = np.linalg.norm((pose_2d[L_SH, :2] - pose_2d[R_SH, :2]) * [w, h])
        hands_2d, hand_of = [], {}
        for i, hl in enumerate(hres.hand_landmarks or []):
            h2 = np.array([[p.x, p.y] for p in hl])
            best, best_d = None, 0.35 * max(sh_px, 1.0)
            for side in ("right", "left"):
                wr = pose_2d[ARM_IDX[side][2], :2]
                d = np.linalg.norm((h2[H_WRIST] - wr) * [w, h])
                if d < best_d:
                    best, best_d = side, d
            hands_2d.append((h2, best))
            if best is not None and best not in hand_of:
                hand_of[best] = i

        for side, (i_s, i_e, i_w) in ARM_IDX.items():
            ob = arms[side]
            ob.s, ob.e, ob.w = to_body(W[i_s]), to_body(W[i_e]), to_body(W[i_w])
            ob.conf["upper"] = float(min(vis[i_s], vis[i_e]))
            ob.conf["fore"] = float(min(vis[i_e], vis[i_w]))
            if side in hand_of:
                hw = np.array([[p.x, p.y, p.z] for p in hres.hand_world_landmarks[hand_of[side]]])
                ob.H = R_body.T @ hand_frame(hw)
                hand_len = np.linalg.norm(hw[H_MIDDLE_MCP] - hw[H_WRIST])
                ob.grip = float(np.linalg.norm(hw[H_THUMB_TIP] - hw[H_INDEX_TIP]) / max(hand_len, 1e-6))
                score = hres.handedness[hand_of[side]][0].score if hres.handedness else 1.0
                ob.conf["hand"] = float(min(score, vis[i_w]))
        return Frame(arms, pose_2d, hands_2d, R_body, t)
