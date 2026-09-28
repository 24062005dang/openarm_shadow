"""Vẽ bằng OpenCV: khung xương người trên ảnh camera + hình que robot (nhìn trước và nhìn ngang)."""
from __future__ import annotations

import unicodedata

import cv2
import numpy as np

POSE_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
              (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
              (17, 18), (18, 19), (19, 20)]
COL = {"right": (255, 140, 0), "left": (0, 140, 255)}   # BGR: phải = xanh dương, trái = cam


def ascii_text(s: str) -> str:
    """cv2.putText chỉ vẽ được ASCII: bỏ dấu tiếng Việt (vd trạng thái SafetyGate) để không hiện '???'."""
    s = s.replace("đ", "d").replace("Đ", "D")
    s = "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))
    return s.encode("ascii", "replace").decode()


def draw_human(img, frame):
    h, w = img.shape[:2]
    th = max(2, w // 250)           # nét dày theo kích thước ảnh (1280 px -> 5 px)
    if frame.pose_2d is not None:
        P = frame.pose_2d
        for a, b in POSE_EDGES:
            if min(P[a, 2], P[b, 2]) > 0.3:
                cv2.line(img, (int(P[a, 0] * w), int(P[a, 1] * h)), (int(P[b, 0] * w), int(P[b, 1] * h)),
                         (0, 220, 0), th)
        for i in (11, 12, 13, 14, 15, 16):
            if P[i, 2] > 0.3:
                cv2.circle(img, (int(P[i, 0] * w), int(P[i, 1] * h)), th + 3, (0, 255, 255), -1)
    for h2, side in frame.hands_2d:
        c = COL.get(side, (160, 160, 160))
        for a, b in HAND_EDGES:
            cv2.line(img, (int(h2[a, 0] * w), int(h2[a, 1] * h)), (int(h2[b, 0] * w), int(h2[b, 1] * h)), c,
                     max(1, th - 2))
    return img


def draw_robot(kins, q: dict, size=(480, 480), q_target: dict | None = None, title="",
               q_meas: dict | None = None):
    """Hai hình chiếu: trái = nhìn từ phía trước robot (ngang = y), phải = nhìn từ bên phải (ngang = x).

    q: lệnh (nét đậm màu) · q_target: mục tiêu (nét mảnh xám) · q_meas: góc đo từ robot thật (nét xanh lá)."""
    W, H = size
    canvas = np.full((H, W, 3), 245, np.uint8)
    half = W // 2
    # Tầm với mỗi tay ~0.55 m quanh vai (vai ở y = ±0.15, z = 0.7): mỗi hình chiếu phải chứa y trong ±0.75 m,
    # z trong 0.1..1.3 m, để tay dang ngang hoặc giơ lên đầu không bị cắt.
    scale = min(half / 1.5, H / 1.4)

    def proj(p, view):
        horiz = p[1] if view == 0 else p[0]     # nhìn từ phía trước robot: tay phải robot ở bên trái ảnh
        cx = half // 2 + view * half
        return int(cx + horiz * scale), int(H * 0.45 + (0.7 - p[2]) * scale)

    for view in (0, 1):
        cv2.putText(canvas, "truoc" if view == 0 else "ben phai", (view * half + 8, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1)
        for s, kin in kins.items():
            for qq, thick, col in ((q_target, 1, (180, 180, 180)), (q, 3, COL[s]), (q_meas, 2, (0, 170, 0))):
                if qq is None or s not in qq or not np.all(np.isfinite(qq[s][:7])):
                    continue
                P = kin.joint_positions(qq[s][:7])
                pts = [kin.p_base] + [P[i] for i in (1, 3, 5, 8)]
                for a, b in zip(pts[:-1], pts[1:]):
                    cv2.line(canvas, proj(a, view), proj(b, view), col, thick)
                for p in pts[1:4]:
                    cv2.circle(canvas, proj(p, view), 4 if thick > 1 else 2, col, -1)
    cv2.line(canvas, (half, 0), (half, H), (200, 200, 200), 1)
    if title:
        cv2.putText(canvas, ascii_text(title), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)
    return canvas


def put_lines(img, lines, org=(10, 24), color=(255, 255, 255)):
    x, y = org
    for ln in lines:
        ln = ascii_text(ln)
        cv2.putText(img, ln, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(img, ln, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        y += 22
    return img


def side_by_side(cam, robot):
    h = cam.shape[0]
    r = cv2.resize(robot, (int(robot.shape[1] * h / robot.shape[0]), h))
    return np.hstack([cam, r])
