"""Hiệu chuẩn ngoại tham số nhiều camera bằng bảng ChArUco (dùng bởi scripts/calibrate_cameras.py).

Cách làm (giống ý của caliscope / Pose2Sim `charuco_static`, viết lại gọn):
- Mỗi lần chụp, các camera cùng thấy bảng: solvePnP trên toạ độ đã khử méo -> tư thế bảng trong từng camera.
- Ngoại tham số camera c so với camera tham chiếu 0: R_c0 = R_c R_0^T, t_c0 = t_c - R_c0 t_0.
- Gộp nhiều lần chụp: trung bình hướng (chordal mean) + trung vị vị trí, bỏ lần chụp lệch nhiều.
- Nội tham số: RealSense lấy từ SDK; webcam hiệu chuẩn bằng chính các ảnh bảng (cv2.calibrateCamera).
"""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import orthonormalize


def make_board(bcfg):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, bcfg.get("dictionary", "DICT_4X4_50")))
    board = cv2.aruco.CharucoBoard((int(bcfg["squares_x"]), int(bcfg["squares_y"])),
                                   float(bcfg["square_m"]), float(bcfg["marker_m"]), d)
    return board, cv2.aruco.CharucoDetector(board)


def detect(detector, bgr):
    """-> (ids (N,), pixel (N,2)) các góc ChArUco thấy được, hoặc (None, None)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    corners, ids, _, _ = detector.detectBoard(gray)
    if ids is None or len(ids) == 0:
        return None, None
    return ids.reshape(-1), corners.reshape(-1, 2)


def board_pose(board, cam, ids, px, min_corners=6):
    """Tư thế bảng trong khung camera (R, t) từ các góc thấy được; dùng toạ độ đã khử méo của cam."""
    if ids is None or len(ids) < min_corners:
        return None
    obj = board.getChessboardCorners()[ids].astype(np.float64)
    norm = cam.normalize(px).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, norm.reshape(-1, 1, 2), np.eye(3), None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3)


def relative_extrinsic(pose_ref, pose_c):
    """Tư thế bảng trong camera 0 và camera c -> (R, t) của camera c với khung thế giới = camera 0."""
    R0, t0 = pose_ref
    Rc, tc = pose_c
    R = Rc @ R0.T
    return R, tc - R @ t0


def average_extrinsics(samples, max_dev_deg=3.0):
    """Gộp nhiều ước lượng (R, t): bỏ ước lượng lệch nhiều so với trung bình rồi tính lại. -> (R, t, inliers)."""
    Rs = np.array([s[0] for s in samples])
    ts = np.array([s[1] for s in samples])
    keep = np.ones(len(samples), bool)
    for _ in range(3):
        R = orthonormalize(Rs[keep].sum(axis=0))
        t = np.median(ts[keep], axis=0)
        ang = np.array([np.degrees(np.arccos(np.clip((np.trace(R.T @ Ri) - 1) / 2, -1, 1))) for Ri in Rs])
        dist = np.linalg.norm(ts - t, axis=1)
        new = (ang <= max(max_dev_deg, 2.5 * np.median(ang))) & (dist <= max(0.01, 2.5 * np.median(dist)))
        if new.sum() < max(3, len(samples) // 3) or (new == keep).all():
            break
        keep = new
    R = orthonormalize(Rs[keep].sum(axis=0))
    t = ts[keep].mean(axis=0)
    return R, t, keep


def reprojection_rms_px(board, cam_ref, cam_c, detections):
    """Sai số (px) khi chiếu góc bảng (theo tư thế bảng ở camera 0) sang camera c qua ngoại tham số đã tính."""
    errs = []
    for (ids0, px0), (ids_c, px_c) in detections:
        pose0 = board_pose(board, cam_ref, ids0, px0)
        if pose0 is None or ids_c is None:
            continue
        obj = board.getChessboardCorners()[ids_c]
        Xw = cam_ref.to_world(obj @ pose0[0].T + pose0[1])
        pred = cam_c.project(Xw)
        errs.append(np.linalg.norm(pred - px_c, axis=1))
    if not errs:
        return np.nan
    e = np.concatenate(errs)
    return float(np.sqrt(np.mean(e ** 2)))


def calibrate_intrinsics(board, detections, image_size):
    """Nội tham số cho webcam từ nhiều ảnh bảng: detections = [(ids, px)]. -> (K, dist, rms_px)."""
    obj, img = [], []
    for ids, px in detections:
        if ids is not None and len(ids) >= 8:
            obj.append(board.getChessboardCorners()[ids].astype(np.float32))
            img.append(px.astype(np.float32).reshape(-1, 1, 2))
    if len(obj) < 8:
        raise ValueError("cần ít nhất 8 ảnh bảng rõ để hiệu chuẩn nội tham số webcam")
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, img, tuple(image_size), None, None)
    return K, dist.reshape(-1), float(rms)


# ----------------------------------------------------------------------------------------------------------
# Độ trễ tương đối giữa các camera (scripts/measure_camera_latency.py)
# ----------------------------------------------------------------------------------------------------------
def motion_energy(prev_small, small):
    """Mức chuyển động giữa 2 ảnh xám thu nhỏ (float32): trung bình |hiệu|. Không cần nhận diện, đúng theo thời gian."""
    return float(np.mean(np.abs(small - prev_small)))


def estimate_time_offset(t_ref, x_ref, t, x, max_lag_s=0.3, step_s=0.002):
    """Độ trễ (s) của tín hiệu x so với x_ref theo thời điểm khung đến máy: x(t) ~ x_ref(t - lag).
    lag > 0: camera này trả khung chậm hơn camera tham chiếu lag giây. Tương quan chéo trên lưới đều.
    -> (lag, hệ số tương quan tại đỉnh), hoặc (nan, nan) nếu đoạn chung quá ngắn."""
    t_ref, x_ref, t, x = (np.asarray(v, float) for v in (t_ref, x_ref, t, x))
    t0, t1 = max(t_ref[0], t[0]) + max_lag_s, min(t_ref[-1], t[-1]) - max_lag_s
    if t1 - t0 < 2.0:
        return float("nan"), float("nan")
    grid = np.arange(t0, t1, step_s)
    a = np.interp(grid, t_ref, x_ref)
    a = (a - a.mean()) / (a.std() + 1e-9)
    best = (float("nan"), -np.inf)
    for lag in np.arange(-max_lag_s, max_lag_s + step_s / 2, step_s):
        b = np.interp(grid + lag, t, x)
        c = float(np.mean(a * (b - b.mean()) / (b.std() + 1e-9)))
        if c > best[1]:
            best = (float(lag), c)
    return best
