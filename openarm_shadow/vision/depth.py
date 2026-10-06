"""Depth RealSense: lấy mẫu, đưa landmark bàn tay về thang mét, mặt phẳng lòng bàn tay, hệ trục bàn tay."""
from __future__ import annotations

import cv2
import numpy as np

from openarm_shadow.core.geometry import unit
from openarm_shadow.vision.landmarks import FINGER_CHAINS, H_INDEX_MCP, H_MIDDLE_MCP, H_PINKY_MCP, H_WRIST


def _intrinsic(intr, name):
    return float(intr[name] if isinstance(intr, dict) else getattr(intr, name))


def sample_depth_with_confidence(depth_m, u, v, radius=3, min_m=0.3, max_m=6.0,
                                 max_delta_m=0.15):
    """Trả về (depth median, confidence) của cụm depth gần pixel landmark."""
    if depth_m is None or depth_m.ndim != 2 or not np.isfinite(u + v):
        return None, 0.0
    h, w = depth_m.shape
    x, y = int(round(u)), int(round(v))
    if not (0 <= x < w and 0 <= y < h):
        return None, 0.0
    r = max(0, int(radius))
    x0, x1, y0, y1 = max(0, x - r), min(w, x + r + 1), max(0, y - r), min(h, y + r + 1)
    patch = np.asarray(depth_m[y0:y1, x0:x1], float)
    valid = np.isfinite(patch) & (patch >= min_m) & (patch <= max_m)
    if not valid.any():
        return None, 0.0
    center = float(depth_m[y, x])
    center_valid = np.isfinite(center) and min_m <= center <= max_m
    if not center_valid:
        yy, xx = np.nonzero(valid)
        nearest = np.argmin((xx + x0 - x) ** 2 + (yy + y0 - y) ** 2)
        center = float(patch[yy[nearest], xx[nearest]])
    cluster = patch[valid & (np.abs(patch - center) <= max_delta_m)]
    if not cluster.size:
        return center, 0.15
    z = float(np.median(cluster))
    mad = float(np.median(np.abs(cluster - z)))
    support = cluster.size / max(patch.size, 1)
    confidence = np.sqrt(support) * np.exp(-mad / 0.02) * (1.0 if center_valid else 0.75)
    return z, float(np.clip(confidence, 0.0, 1.0))


def sample_depth(depth_m, u, v, radius=3, min_m=0.3, max_m=6.0, max_delta_m=0.15):
    """API cũ: chỉ trả depth để các caller body-pose và test hiện tại không đổi."""
    return sample_depth_with_confidence(depth_m, u, v, radius, min_m, max_m, max_delta_m)[0]


def deproject_pixel(intr, u, v, depth_m):
    """Pixel RGB + depth (m) -> điểm [x phải, y xuống, z trước] trong camera."""
    if not isinstance(intr, dict):
        try:
            import pyrealsense2 as rs
            return np.asarray(rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], float(depth_m)),
                              dtype=float)
        except (ImportError, RuntimeError, TypeError, ValueError):
            # Fallback pinhole giúp test/offline intrinsics vẫn dùng được; D455 thật đi qua SDK ở trên.
            pass
    fx, fy = _intrinsic(intr, "fx"), _intrinsic(intr, "fy")
    ppx, ppy = _intrinsic(intr, "ppx"), _intrinsic(intr, "ppy")
    z = float(depth_m)
    return np.array([(u - ppx) * z / fx, (v - ppy) * z / fy, z])


def fit_metric_depth(relative_z, measured_z, confidence=None, previous=None, min_points=4):
    """Fit z_metric = a*z_mediapipe + b; trả (a,b) hoặc previous khi dữ liệu chưa đủ.

    MediaPipe cho hình dạng depth tương đối. D455 cung cấp scale/offset metric. Fit robust này cho phép
    khôi phục landmark nằm trên lỗ stereo mà vẫn giữ mọi điểm trong camera frame của D455.
    """
    rz = np.asarray(relative_z, float)
    mz = np.asarray(measured_z, float)
    cw = np.ones_like(rz) if confidence is None else np.asarray(confidence, float)
    valid = np.isfinite(rz) & np.isfinite(mz) & (cw > 0.05)
    model = None
    if np.count_nonzero(valid) >= int(min_points) and np.ptp(rz[valid]) > 1e-3:
        X = np.column_stack([rz[valid], np.ones(np.count_nonzero(valid))])
        y, w = mz[valid], np.sqrt(cw[valid])
        beta = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]
        residual = y - X @ beta
        med = np.median(residual)
        mad = np.median(np.abs(residual - med))
        keep = np.abs(residual - med) <= max(0.015, 3.0 * mad)
        if np.count_nonzero(keep) >= int(min_points):
            beta = np.linalg.lstsq(X[keep] * w[keep, None], y[keep] * w[keep], rcond=None)[0]
        a, b = map(float, beta)
        if 0.03 <= a <= 3.0 and np.isfinite(a + b):
            model = (a, b)
    # GMH-D style anchor: z của wrist (index 0) đặt scale metric ban đầu khi chưa fit đủ điểm.
    if model is None and len(mz) and np.isfinite(mz[0]):
        a = float(np.clip(mz[0], 0.03, 3.0))
        model = (a, float(mz[0] - a * rz[0]))
    if model is None:
        return previous
    if previous is not None and np.all(np.isfinite(previous)):
        # Model đổi chậm hơn từng pixel depth, tránh toàn bộ bàn tay co/giãn khi số điểm hợp lệ thay đổi.
        model = tuple(0.7 * float(p) + 0.3 * float(n) for p, n in zip(previous, model))
    return model


def fuse_hand_landmarks(hand_landmarks, depth_m, intr, cfg, previous_model=None):
    """MediaPipe 2D + D455 depth -> 21 điểm metric cùng D455 camera frame và diagnostics."""
    h, w = depth_m.shape
    uv = np.array([[p.x * w, p.y * h] for p in hand_landmarks], dtype=float)
    relative_z = np.array([p.z for p in hand_landmarks], dtype=float)
    measured = np.full(len(hand_landmarks), np.nan)
    direct_conf = np.zeros(len(hand_landmarks), dtype=float)
    radius = int(cfg.get("hand_patch_radius", 2))
    for i, (u, v) in enumerate(uv):
        measured[i], direct_conf[i] = sample_depth_with_confidence(
            depth_m, u, v, radius=radius, min_m=cfg.get("min_depth_m", 0.3),
            max_m=cfg.get("max_depth_m", 6.0), max_delta_m=cfg.get("max_local_delta_m", 0.15))
    min_points = int(cfg.get("hand_min_direct_points", 4))
    model = fit_metric_depth(relative_z, measured, direct_conf, previous_model, min_points)
    predicted = np.full_like(measured, np.nan)
    if model is not None:
        predicted = model[0] * relative_z + model[1]
    fused_z = predicted.copy()
    direct = np.isfinite(measured)
    only_direct = direct & ~np.isfinite(fused_z)
    fused_z[only_direct] = measured[only_direct]
    both = direct & np.isfinite(predicted)
    fused_z[both] = direct_conf[both] * measured[both] + (1.0 - direct_conf[both]) * predicted[both]
    points = np.full((len(hand_landmarks), 3), np.nan)
    for i, z in enumerate(fused_z):
        if np.isfinite(z) and cfg.get("min_depth_m", 0.3) <= z <= cfg.get("max_depth_m", 6.0):
            points[i] = deproject_pixel(intr, uv[i, 0], uv[i, 1], z)
    n_direct = int(np.count_nonzero(direct))
    n_fused = int(np.count_nonzero(np.all(np.isfinite(points), axis=1)))
    mean_conf = float(np.mean(direct_conf[direct])) if n_direct else 0.0
    mode = "DEPTH" if n_direct == len(hand_landmarks) else ("FUSED" if n_fused else "NONE")
    info = {"direct": n_direct, "fused": n_fused, "confidence": mean_conf, "mode": mode,
            "model": model}
    return points, info


def project_point(intr, point):
    """Điểm camera-frame -> pixel RGB, ưu tiên projection có distortion của librealsense."""
    p = np.asarray(point, float)
    if not np.all(np.isfinite(p)) or p[2] <= 0:
        return None
    if not isinstance(intr, dict):
        try:
            import pyrealsense2 as rs
            return np.asarray(rs.rs2_project_point_to_pixel(intr, p.tolist()), dtype=float)
        except (ImportError, RuntimeError, TypeError, ValueError):
            pass
    fx, fy = _intrinsic(intr, "fx"), _intrinsic(intr, "fy")
    ppx, ppy = _intrinsic(intr, "ppx"), _intrinsic(intr, "ppy")
    return np.array([p[0] * fx / p[2] + ppx, p[1] * fy / p[2] + ppy])


def open_finger_count(points):
    """Đếm bốn ngón chính đang duỗi từ hình học 3D; không dùng thumb vì biến thiên lớn."""
    p = np.asarray(points, float)
    if p.shape != (21, 3):
        return 0
    count = 0
    for mcp, pip, dip, tip in FINGER_CHAINS:
        ids = (H_WRIST, mcp, pip, dip, tip)
        if not np.all(np.isfinite(p[list(ids)])):
            continue
        chain = (np.linalg.norm(p[pip] - p[mcp]) + np.linalg.norm(p[dip] - p[pip]) +
                 np.linalg.norm(p[tip] - p[dip]))
        straight = np.linalg.norm(p[tip] - p[mcp]) / max(chain, 1e-6)
        reach = np.linalg.norm(p[tip] - p[H_WRIST]) / max(np.linalg.norm(p[mcp] - p[H_WRIST]), 1e-6)
        count += int(straight > 0.78 and reach > 1.30)
    return count


def fit_palm_plane(depth_m, intr, hand_landmarks, cfg):
    """Fit plane robust trong polygon lòng bàn tay; trả normal, center và diagnostics."""
    h, w = depth_m.shape
    uv = np.array([[p.x * w, p.y * h] for p in hand_landmarks], dtype=float)
    palm_ids = [H_WRIST, H_INDEX_MCP, H_MIDDLE_MCP, 13, H_PINKY_MCP]
    poly = cv2.convexHull(np.round(uv[palm_ids]).astype(np.int32))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, poly, 1)
    area = int(mask.sum())
    if area < 20:
        return None, None, {"inliers": 0, "rms_m": float("inf")}
    erode_px = max(1, int(np.sqrt(area) * 0.08))
    kernel = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)
    mask = cv2.erode(mask, kernel)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None, None, {"inliers": 0, "rms_m": float("inf")}
    z = np.asarray(depth_m[ys, xs], float)
    valid = np.isfinite(z) & (z >= cfg.get("min_depth_m", 0.3)) & (z <= cfg.get("max_depth_m", 6.0))
    xs, ys, z = xs[valid], ys[valid], z[valid]
    if not len(z):
        return None, None, {"inliers": 0, "rms_m": float("inf")}
    med = float(np.median(z))
    band = float(cfg.get("palm_depth_band_m", 0.06))
    keep = np.abs(z - med) <= band
    xs, ys, z = xs[keep], ys[keep], z[keep]
    # Giới hạn chi phí deproject nhưng vẫn phủ đều ROI.
    if len(z) > 400:
        take = np.linspace(0, len(z) - 1, 400).astype(int)
        xs, ys, z = xs[take], ys[take], z[take]
    min_points = int(cfg.get("palm_min_points", 35))
    if len(z) < min_points:
        return None, None, {"inliers": int(len(z)), "rms_m": float("inf")}
    pts = np.array([deproject_pixel(intr, x, y, zz) for x, y, zz in zip(xs, ys, z)])
    center = np.mean(pts, axis=0)
    _, _, vt = np.linalg.svd(pts - center, full_matrices=False)
    normal = unit(vt[-1])
    residual = np.abs((pts - center) @ normal)
    robust = residual <= max(0.008, 3.0 * float(np.median(residual)))
    if np.count_nonzero(robust) >= min_points:
        pts = pts[robust]
        center = np.mean(pts, axis=0)
        _, _, vt = np.linalg.svd(pts - center, full_matrices=False)
        normal = unit(vt[-1])
    rms = float(np.sqrt(np.mean(((pts - center) @ normal) ** 2)))
    if rms > float(cfg.get("palm_plane_max_rms_m", 0.012)):
        return None, center, {"inliers": int(len(pts)), "rms_m": rms}
    return normal, center, {"inliers": int(len(pts)), "rms_m": rms}


def palm_frame_from_depth(points, plane_normal=None, plane_quality=0.0, side=None):
    """Frame camera của tay: x hướng ngón, y út->trỏ, z=x×y; plane chỉ tinh chỉnh pháp tuyến."""
    p = np.asarray(points, float)
    ids = [H_WRIST, H_INDEX_MCP, H_MIDDLE_MCP, 13, H_PINKY_MCP]
    if p.shape != (21, 3) or not np.all(np.isfinite(p[ids])):
        return None, None
    center = np.mean(p[ids], axis=0)
    x = unit(0.5 * (p[H_MIDDLE_MCP] + p[13]) - p[H_WRIST])
    across = p[H_INDEX_MCP] - p[H_PINKY_MCP]
    y_land = unit(across - (across @ x) * x)
    z_land = unit(np.cross(x, y_land))
    # Hai bàn tay có chirality đối nhau. Bù dấu để z đều hướng ra khỏi lòng bàn tay.
    if side == "right":
        z_land = -z_land
    z = z_land
    if plane_normal is not None and np.linalg.norm(plane_normal) > 0.5:
        pn = unit(plane_normal)
        if pn @ z_land < 0:
            pn = -pn
        weight = float(np.clip(plane_quality, 0.0, 0.85))
        z = unit((1.0 - weight) * z_land + weight * pn)
    y = unit(np.cross(z, x))
    x = unit(np.cross(y, z))
    R = np.column_stack([x, y, z])
    return R, center
