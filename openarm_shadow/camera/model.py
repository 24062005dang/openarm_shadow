"""Mô hình camera (ngoại tham số, nội tham số, chiếu / chuẩn hoá) và đọc file hiệu chuẩn."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml


# ----------------------------------------------------------------------------------------------------------
# Mô hình camera
# ----------------------------------------------------------------------------------------------------------
@dataclass
class CameraModel:
    """Camera trong khung thế giới: X_cam = R @ X_world + t. Nội tham số: intrinsics RealSense hoặc K + dist."""
    name: str
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))
    K: np.ndarray | None = None
    dist: np.ndarray | None = None
    rs_intr: object = None
    size: tuple | None = None          # (w, h) lúc hiệu chuẩn nội tham số webcam

    def set_realsense_intrinsics(self, intr):
        self.rs_intr = intr
        if isinstance(intr, dict):
            self.K = np.array([[intr["fx"], 0, intr["ppx"]], [0, intr["fy"], intr["ppy"]], [0, 0, 1.0]])
            self.dist = np.zeros(5)
        else:
            self.K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1.0]])
            self.dist = np.asarray(intr.coeffs, float)

    @property
    def fx(self):
        return float(self.K[0, 0]) if self.K is not None else 600.0

    @property
    def center(self):
        return -self.R.T @ self.t

    def to_world(self, Xc):
        return (np.asarray(Xc, float) - self.t) @ self.R

    def to_cam(self, Xw):
        return np.asarray(Xw, float) @ self.R.T + self.t

    def normalize(self, uv):
        """Pixel (N,2) -> toạ độ chuẩn hoá đã khử méo (x/z, y/z). Hàng NaN giữ NaN."""
        uv = np.asarray(uv, float).reshape(-1, 2)
        out = np.full_like(uv, np.nan)
        ok = np.all(np.isfinite(uv), axis=1)
        if not ok.any():
            return out
        if self.rs_intr is not None and not isinstance(self.rs_intr, dict):
            try:
                import pyrealsense2 as rs
                for i in np.flatnonzero(ok):
                    p = rs.rs2_deproject_pixel_to_point(self.rs_intr, [float(uv[i, 0]), float(uv[i, 1])], 1.0)
                    out[i] = [p[0] / p[2], p[1] / p[2]]
                return out
            except (ImportError, RuntimeError, TypeError, ValueError):
                pass
        if self.K is None:
            raise RuntimeError(f"camera {self.name}: chưa có nội tham số")
        d = np.zeros(5) if self.dist is None else np.asarray(self.dist, float)
        out[ok] = cv2.undistortPoints(uv[ok].reshape(-1, 1, 2), self.K, d).reshape(-1, 2)
        return out

    def project(self, Xw):
        """Điểm thế giới (N,3) -> pixel (N,2); điểm sau camera -> NaN."""
        Xc = self.to_cam(np.asarray(Xw, float).reshape(-1, 3))
        out = np.full((len(Xc), 2), np.nan)
        ok = np.all(np.isfinite(Xc), axis=1) & (Xc[:, 2] > 1e-6)
        if not ok.any():
            return out
        if self.rs_intr is not None and not isinstance(self.rs_intr, dict):
            try:
                import pyrealsense2 as rs
                for i in np.flatnonzero(ok):
                    out[i] = rs.rs2_project_point_to_pixel(self.rs_intr, Xc[i].tolist())
                return out
            except (ImportError, RuntimeError, TypeError, ValueError):
                pass
        d = np.zeros(5) if self.dist is None else np.asarray(self.dist, float)
        px, _ = cv2.projectPoints(Xc[ok].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), self.K, d)
        out[ok] = px.reshape(-1, 2)
        return out


def load_calibration(path, names):
    """Đọc file hiệu chuẩn (scripts/calibrate_cameras.py) -> list CameraModel theo thứ tự `names`."""
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"Chưa có file hiệu chuẩn camera {p}.\n"
                         "Chạy: python scripts/calibrate_cameras.py --config config/fusion_2cam.yaml")
    data = yaml.safe_load(p.read_text())
    cams = []
    for n in names:
        if n not in data.get("cameras", {}) or "R" not in data["cameras"][n]:
            raise SystemExit(f"File {p} chưa có ngoại tham số camera '{n}'. Hiệu chuẩn: "
                             "python scripts/calibrate_cameras.py --config <config fusion đang dùng>")
        c = data["cameras"][n]
        cm = CameraModel(n, np.asarray(c["R"], float), np.asarray(c["t"], float))
        if c.get("K") is not None:
            cm.K, cm.dist = np.asarray(c["K"], float), np.asarray(c.get("dist") or np.zeros(5), float)
        cm.size = tuple(c["size"]) if c.get("size") else None
        cams.append(cm)
    return cams


def check_image_size(cam, bgr):
    """Nội tham số webcam chỉ đúng ở đúng độ phân giải lúc hiệu chuẩn. Lệch -> dừng, không chạy với số sai."""
    size = getattr(cam, "size", None)
    if cam.rs_intr is not None or not size:
        return
    got = (int(bgr.shape[1]), int(bgr.shape[0]))
    if got != tuple(int(v) for v in size):
        raise SystemExit(f"Camera '{cam.name}' đang cho ảnh {got[0]}x{got[1]} nhưng được hiệu chuẩn ở "
                         f"{size[0]}x{size[1]}.\nĐặt camera.width/height trong config cho khớp, hoặc hiệu chuẩn lại.")
