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


def pixel(p, img, margin=2.0):
    """Điểm ảnh (x, y) float -> tuple int cho cv2, hoặc None nếu NaN hay xa ngoài ảnh quá margin x kích thước ảnh.

    Điểm 3D chiếu lại (fusion) nằm sát mặt phẳng camera cho toạ độ hàng tỉ pixel: int() vượt giới hạn int của
    OpenCV -> cv2.circle báo "Can't parse 'center'" và làm dừng cả chương trình. Điểm như vậy chỉ bỏ qua khi vẽ."""
    p = np.asarray(p, float)
    lim = margin * max(img.shape[:2])
    if p.shape[-1] < 2 or not np.all(np.isfinite(p[:2])) or np.any(np.abs(p[:2]) > lim):
        return None
    return int(round(p[0])), int(round(p[1]))


TORSO_EDGES = {(11, 12), (11, 23), (12, 24), (23, 24)}


def draw_human(img, frame, skip=(), torso_estimated=False):
    """skip: chỉ số landmark Pose KHÔNG vẽ (cùng các đoạn nối tới nó), vd vai/hông khi thân được vẽ riêng (fusion).
    torso_estimated: 1 camera đang dùng hướng thân tham chiếu (tay che thân, body_ref) -> vẽ thân thô màu cam + chữ,
    để biết khung thân trên ảnh lúc đó KHÔNG được dùng."""
    h, w = img.shape[:2]
    th = max(2, w // 250)           # nét dày theo kích thước ảnh (1280 px -> 5 px)
    if frame.pose_2d is not None:
        P = frame.pose_2d
        for a, b in POSE_EDGES:
            if a in skip or b in skip:
                continue
            if min(P[a, 2], P[b, 2]) > 0.3:
                col = (0, 140, 255) if torso_estimated and (a, b) in TORSO_EDGES else (0, 220, 0)
                cv2.line(img, (int(P[a, 0] * w), int(P[a, 1] * h)), (int(P[b, 0] * w), int(P[b, 1] * h)),
                         col, th)
        for i in (11, 12, 13, 14, 15, 16):
            if i not in skip and P[i, 2] > 0.3:
                cv2.circle(img, (int(P[i, 0] * w), int(P[i, 1] * h)), th + 3, (0, 255, 255), -1)
    for h2, side in frame.hands_2d:
        c = COL.get(side, (160, 160, 160))
        for a, b in HAND_EDGES:
            cv2.line(img, (int(h2[a, 0] * w), int(h2[a, 1] * h)), (int(h2[b, 0] * w), int(h2[b, 1] * h)), c,
                     max(1, th - 2))
    # Palm frame metric từ D455: x đỏ (hướng ngón), y xanh lá (út->trỏ), z xanh dương (pháp tuyến).
    axis_colors = ((0, 0, 255), (0, 220, 0), (255, 0, 0))
    for ob in frame.arms.values():
        if ob.hand_axes_px is None:
            continue
        origin = pixel(ob.hand_axes_px[0], img)
        if origin is None:
            continue
        cv2.circle(img, origin, th + 2, (255, 255, 255), -1)
        for endpoint, color in zip(ob.hand_axes_px[1:], axis_colors):
            end = pixel(endpoint, img)
            if end is not None:
                cv2.arrowedLine(img, origin, end, color, max(2, th - 1), tipLength=0.2)
    return img


def draw_robot(kins, q: dict, size=(480, 480), q_target: dict | None = None, title="",
               q_meas: dict | None = None, zoom_to_arms=False):
    """Hai hình chiếu: trái = nhìn từ phía trước robot (ngang = y), phải = nhìn từ bên phải (ngang = x).

    q: lệnh (nét đậm màu) · q_target: mục tiêu (nét mảnh xám) · q_meas: góc đo từ robot thật (nét xanh lá).
    zoom_to_arms: phóng to, căn giữa quanh vai các tay đang vẽ (chỉ chừa tầm với ~0.6 m) thay vì cả thân robot."""
    W, H = size
    canvas = np.full((H, W, 3), 245, np.uint8)
    half = W // 2
    # Tầm với mỗi tay ~0.55 m quanh vai (vai ở y = ±0.15, z = 0.7): mỗi hình chiếu phải chứa y trong ±0.75 m,
    # z trong 0.1..1.3 m, để tay dang ngang hoặc giơ lên đầu không bị cắt.
    scale = min(half / 1.5, H / 1.4)
    cen = [0.0, 0.0]                             # tâm ngang của hình chiếu trước (y) / bên phải (x)
    z0, zc = 0.7, 0.45                           # độ cao vai (m) đặt ở zc * H
    if zoom_to_arms and kins:
        reach = 0.6
        sh = np.array([k.display_keypoints(np.zeros(7))["shoulder"] for k in kins.values()])
        spans = []
        for view, ax in ((0, 1), (1, 0)):
            lo, hi = sh[:, ax].min() - reach, sh[:, ax].max() + reach
            cen[view] = 0.5 * (lo + hi)
            spans.append(hi - lo)
        z0, zc = float(sh[:, 2].mean()), 0.5
        scale = min(half / max(spans), H / (2 * reach))

    def proj(p, view):
        horiz = p[1] if view == 0 else p[0]     # nhìn từ phía trước robot: tay phải robot ở bên trái ảnh
        cx = half // 2 + view * half
        return int(cx + (horiz - cen[view]) * scale), int(H * zc + (z0 - p[2]) * scale)

    def draw_frame(origin, R, view, label, length=0.075):
        colors = ((0, 0, 255), (0, 180, 0), (255, 0, 0))  # x đỏ, y xanh lá, z xanh dương
        po = proj(origin, view)
        cv2.circle(canvas, po, 3, (30, 30, 30), -1)
        for j, color in enumerate(colors):
            cv2.arrowedLine(canvas, po, proj(origin + R[:, j] * length, view), color, 2, tipLength=0.22)
        cv2.putText(canvas, label, (po[0] + 4, po[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 40, 40), 1)

    for view in (0, 1):
        cv2.putText(canvas, "truoc" if view == 0 else "ben phai", (view * half + 8, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1)
        for s, kin in kins.items():
            for qq, thick, col in ((q_target, 1, (180, 180, 180)), (q, 3, COL[s]), (q_meas, 2, (0, 170, 0))):
                if qq is None or s not in qq or not np.all(np.isfinite(qq[s][:7])):
                    continue
                k = kin.display_keypoints(qq[s][:7])
                pts = [kin.p_base, k["shoulder"], k["elbow"], k["wrist"], k["tool"]]
                for a, b in zip(pts[:-1], pts[1:]):
                    cv2.line(canvas, proj(a, view), proj(b, view), col, thick)
                for p in pts[1:4]:
                    cv2.circle(canvas, proj(p, view), 4 if thick > 1 else 2, col, -1)
            # Chỉ vẽ frame của lệnh hiện tại để không chồng ba bộ target/cmd/measured.
            if q is not None and s in q and np.all(np.isfinite(q[s][:7])):
                qs = q[s][:7]
                k = kin.display_keypoints(qs)
                draw_frame(0.5 * (k["shoulder"] + k["elbow"]), kin.R0(qs, 3), view, "U")
                draw_frame(0.5 * (k["elbow"] + k["wrist"]), kin.R0(qs, 5), view, "F")
                draw_frame(0.5 * (k["wrist"] + k["tool"]), kin.R_tool(qs), view, "H")
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


def compose_view(cam, render_robot, layout="auto", robot_height=480):
    """Ghép ảnh camera với hình robot mô phỏng. render_robot(size, zoom) -> ảnh robot cỡ size=(W, H)
    (zoom: phóng to quanh vai, xem draw_robot).

    layout "right": robot bên phải camera, cao bằng camera (cũ). "below": robot nằm dưới, rộng bằng cả dải camera,
    cao robot_height -> hình robot to hơn nhiều khi có 2 camera (dải camera rộng và thấp). "auto": dải camera rộng
    hơn 2 lần chiều cao (nhiều camera) thì "below", còn lại "right"."""
    h, w = cam.shape[:2]
    if layout == "auto":
        layout = "below" if w > 2 * h else "right"
    if layout == "below":
        return np.vstack([cam, render_robot((w, int(robot_height)), True)])
    return side_by_side(cam, render_robot((480, 480), False))
