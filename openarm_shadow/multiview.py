"""Fusion nhiều camera: mỗi camera chạy MediaPipe riêng -> triangulation có trọng số -> ArmObs như 1 camera.

Đầu ra giống hệt Perception (Frame với ArmObs vai/khuỷu/cổ tay/hướng bàn tay trong khung thân), nên retarget,
bộ lọc và SafetyGate giữ nguyên. Khung thế giới = khung camera tham chiếu (camera đầu tiên trong
fusion.cameras, thường là camera trực diện): x phải, y xuống, z ra xa camera.

Ý tưởng lấy từ (viết lại, không chép code):
- Pose2Sim (BSD-3, common.weighted_triangulation): DLT với mỗi hàng nhân độ tin cậy 2D; loại camera có sai số
  chiếu lại lớn (triangulation.triangulation_from_best_cameras).
- aniposelib / caliscope (BSD-2): mô hình camera nhóm, ngoại tham số hiệu chuẩn bằng bảng ChArUco.
- stereohand (MIT): mỗi camera một luồng chụp, ghép khung theo thời điểm.
- Fortini et al. 2023 (trọng số theo góc nhìn) và AnyTeleop 2023 (ưu tiên camera tin cậy nhất): trọng số của
  một camera cho các điểm bàn tay tăng khi camera đó nhìn thẳng vào lòng/lưng bàn tay.
- ATHENA (MIT, Mulla & Michaels 2025): bàn tay ở một camera phải nằm gần cổ tay Pose đã triangulate chiếu vào
  camera đó, không thì bỏ bàn tay ở camera đó (tay người khác, gán nhầm tay). Xem thêm handfusion.py.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from .geometry import unit
from .handfusion import HandShape, OrientationFusion, PalmModel, hand_forearm_angle
from .perception import (ArmObs, Frame, H_INDEX_MCP, H_INDEX_TIP, H_MIDDLE_MCP, H_PINKY_MCP, H_THUMB_TIP,
                         H_WRIST, L_EL, L_HIP, L_SH, L_WR, R_EL, R_HIP, R_SH, R_WR, ARM_IDX, body_frame,
                         open_finger_count, palm_frame_from_depth, rotation_distance, sample_depth,
                         slerp_rotation)

BODY_IDS = (L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP)
PALM_IDS = (H_WRIST, H_INDEX_MCP, H_MIDDLE_MCP, 13, H_PINKY_MCP)
REALSENSE_NAMES = {"realsense", "rs", "d455", "d435", "d435i"}
FLIP_X = np.diag([1.0, -1.0, -1.0])        # quay 180° quanh trục ngón tay (x của khung bàn tay)


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
        if n not in data.get("cameras", {}):
            raise SystemExit(f"File {p} không có camera '{n}'. Hiệu chuẩn lại.")
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


# ----------------------------------------------------------------------------------------------------------
# Triangulation có trọng số
# ----------------------------------------------------------------------------------------------------------
def triangulate_weighted(obs, depth=()):
    """Bình phương tối thiểu có trọng số cho 1 điểm.

    obs: [(CameraModel, (x, y) chuẩn hoá, w)], depth: [(X_world, w)]. Mỗi camera cho 2 phương trình tuyến tính
    (dạng DLT không đồng nhất); mỗi điểm depth cho 3 phương trình X = X_depth. Trả None nếu thiếu ràng buộc.
    """
    rows, rhs = [], []
    for cam, (x, y), w in obs:
        R, t = cam.R, cam.t
        rows += [w * (x * R[2] - R[0]), w * (y * R[2] - R[1])]
        rhs += [w * (t[0] - x * t[2]), w * (t[1] - y * t[2])]
    for Xd, w in depth:
        rows += list(w * np.eye(3))
        rhs += list(w * np.asarray(Xd, float))
    if len(rows) < 3:
        return None
    A, b = np.asarray(rows), np.asarray(rhs)
    if np.linalg.matrix_rank(A, tol=1e-9) < 3:
        return None
    return np.linalg.lstsq(A, b, rcond=None)[0]


REF_FX = 600.0    # tiêu cự của ảnh 640x480 thường gặp: sai số được quy về thang này


def reprojection_px(cam, X, xy):
    """Sai số chiếu lại, tính bằng px QUY ĐỔI về ảnh 640x480 (tiêu cự 600 px) để ngưỡng reproj_thresh_px không
    phụ thuộc độ phân giải: webcam 1280x720 và D435i 640x480 dùng chung một ngưỡng (1 px ~ 0,1°)."""
    Xc = cam.to_cam(X)
    if Xc[2] <= 1e-6:
        return np.inf
    return float(np.hypot(Xc[0] / Xc[2] - xy[0], Xc[1] / Xc[2] - xy[1]) * REF_FX)


def fuse_point(obs, depth, reproj_px=25.0, depth_weight=0.3, depth_tol_m=0.04):
    """Hợp nhất 1 điểm từ nhiều camera. Trả (X, conf, info).

    1. >= 2 camera: triangulate 2D; nếu sai số chiếu lại vượt ngưỡng thì bỏ camera tệ nhất (còn >= 2 camera),
       hoặc (còn 2 camera) thử "1 camera + depth của chính nó" và giữ phương án chiếu lại tốt nhất.
    2. Depth chỉ được thêm khi khớp với nghiệm 2D trong depth_tol_m. Depth lệch:
       - GẦN camera hơn nghiệm 2D = vật che phía trước (ngón che ngón) -> bỏ depth, giữ nguyên độ tin cậy;
       - XA hơn = mâu thuẫn (lỗi dọc đường epipolar mà 2 camera không tự thấy, hoặc depth rơi vào nền)
         -> info["conflict"] = True, độ tin cậy x0.5.
    3. 1 camera: chỉ dùng được nếu camera đó có depth.

    obs: [(cam, xy, conf)] hoặc [(cam, xy, conf, trust)]. conf (độ tin cậy MediaPipe) quyết định độ tin cậy trả về;
    trust (trọng số camera, kích thước bàn tay, góc nhìn) chỉ dùng để trộn các camera khi triangulate.
    """
    obs = [o for o in obs if o[2] > 0.05 and np.all(np.isfinite(o[1]))]
    conf_of = {id(o[0]): float(o[2]) for o in obs}
    obs = [(o[0], o[1], float(o[2]) * (float(o[3]) if len(o) > 3 else 1.0)) for o in obs]
    dep = {id(o[0]): d for o, d in zip(obs, [None] * len(obs))}
    for cam, Xd, w in depth:
        if np.all(np.isfinite(Xd)) and w > 0.05:
            dep[id(cam)] = (np.asarray(Xd, float), w)
    info = {"views": len(obs), "err_px": np.nan, "depth": 0, "conflict": False}
    if len(obs) >= 2:
        use = list(obs)
        X = triangulate_weighted(use)
        errs = [reprojection_px(c, X, xy) for c, xy, _ in use] if X is not None else []
        while X is not None and max(errs) > reproj_px and len(use) > 2:
            use.pop(int(np.argmax(errs)))
            X = triangulate_weighted(use)
            errs = [reprojection_px(c, X, xy) for c, xy, _ in use] if X is not None else []
        conf = float(np.mean([conf_of[id(c)] for c, _, _ in use]))
        if X is not None and max(errs) > reproj_px:
            # 2 camera mâu thuẫn (một camera đoán sai điểm, vd bị che): depth là bằng chứng độc lập để phân xử.
            # Ứng viên = 1 camera + depth của chính nó; chọn ứng viên khớp nhất với các camera còn lại.
            cands = []
            for c, xy, w in use:
                d = dep.get(id(c))
                if d is None:
                    continue
                Xc = triangulate_weighted([(c, xy, w)], [(d[0], depth_weight * d[1])])
                if Xc is not None:
                    e_other = max([reprojection_px(c2, Xc, xy2) for c2, xy2, _ in use if c2 is not c] or [0.0])
                    cands.append((e_other, len(cands), Xc))
            if cands:
                X = min(cands)[2]
                info["depth"] = 1
                conf *= 0.5
            else:
                conf *= 0.3
        if X is None:
            return None, 0.0, info
        extra = []
        for c, _, _ in use:
            d = dep.get(id(c))
            if d is None:
                continue
            if np.linalg.norm(d[0] - X) < depth_tol_m:
                extra.append((d[0], depth_weight * d[1]))
            elif info["depth"] == 0 and c.to_cam(d[0])[2] > c.to_cam(X)[2]:
                # Depth gần hơn nghiệm 2D = mặt phía trước che điểm (ngón che ngón): bỏ depth, không sao.
                # Depth XA hơn: không thể là vật che -> một camera đoán sai điểm dọc đường epipolar (2 camera
                # không tự phát hiện được) hoặc depth rơi vào nền ở mép tay. Không đoán bên nào: hạ độ tin cậy.
                info["conflict"] = True
                conf *= 0.5
        if extra and info["depth"] == 0:
            X2 = triangulate_weighted(use, extra)
            if X2 is not None:
                X, info["depth"] = X2, len(extra)
        info["err_px"] = float(max(reprojection_px(c, X, xy) for c, xy, _ in use))
        info["views"] = len(use)
        return X, conf, info
    if len(obs) == 1 and dep.get(id(obs[0][0])) is not None:
        d = dep[id(obs[0][0])]
        info["depth"] = 1
        return d[0], 0.8 * min(conf_of[id(obs[0][0])], d[1]), info
    return None, 0.0, info


# ----------------------------------------------------------------------------------------------------------
# Hướng bàn tay: giữ liên tục dấu
# ----------------------------------------------------------------------------------------------------------
class HandOrientationTracker:
    """Làm mượt hướng bàn tay và chống lật úp/ngửa.

    - Khi mọi camera đều nhìn cạnh bàn tay (quality thấp), pháp tuyến có thể lật dấu: chọn trong hai giả thuyết
      R và R quay 180° quanh trục ngón cái nào gần hướng trước hơn.
    - Nhảy quá max_jump_deg trong 1 khung: giữ hướng cũ, chỉ nhận khi lặp lại confirm_frames khung liên tiếp.
    - Mất quan sát: giữ hướng cũ tối đa hold_frames khung rồi mới báo NONE (không tự nhận hướng lật).
    """

    def __init__(self, alpha=0.6, hold_frames=8, ambiguous_quality=0.35, max_jump_deg=100.0, confirm_frames=4,
                 alpha_max=0.85):
        # alpha: tỉ lệ nhận hướng mới khi nhìn KÉM (nhiều làm mượt); alpha_max: khi nhìn rõ (ít trễ, để cổ tay
        # không chậm hơn cánh tay; bộ lọc từng khớp phía sau vẫn làm mượt).
        self.alpha, self.alpha_max, self.hold, self.amb_q = alpha, max(alpha, alpha_max), hold_frames, ambiguous_quality
        self.max_jump, self.confirm = np.deg2rad(max_jump_deg), confirm_frames
        self.R, self.miss, self.jumps = None, 0, 0

    def update(self, R, quality):
        if R is None:
            self.miss += 1
            if self.R is not None and self.miss <= self.hold:
                return self.R, "HOLD"
            self.R = None
            return None, "NONE"
        self.miss = 0
        if self.R is None:
            self.R, self.jumps = R, 0
            return R, "FUSED"
        mode = "FUSED"
        d, R_alt = rotation_distance(self.R, R), R @ FLIP_X
        d_alt = rotation_distance(self.R, R_alt)
        if quality < self.amb_q and d_alt < d:
            R, d, mode = R_alt, d_alt, "SIGN-FIX"
        if d > self.max_jump:
            self.jumps += 1
            if self.jumps < self.confirm:
                return self.R, "HOLD"
        self.jumps = 0
        k = float(np.clip((quality - self.amb_q) / max(1.0 - self.amb_q, 1e-6), 0.0, 1.0))
        self.R = slerp_rotation(self.R, R, self.alpha + (self.alpha_max - self.alpha) * k)
        return self.R, mode


# ----------------------------------------------------------------------------------------------------------
# Nguồn nhiều camera, đồng bộ bằng phần mềm
# ----------------------------------------------------------------------------------------------------------
@dataclass
class MultiSample:
    views: list                 # CameraSample theo thứ tự fusion.cameras
    t: float                    # thời điểm (s, time.monotonic) của khung camera tham chiếu
    skew_s: float = 0.0         # lệch thời gian lớn nhất giữa các khung được ghép (không tính camera stale)
    stale: list = None          # stale[i] = True: camera i không có khung đủ mới (treo, mất kết nối) -> bỏ qua


class MultiCameraSource:
    """Mỗi camera một luồng đọc; read() lấy khung mới nhất của camera tham chiếu và khung gần thời điểm nhất
    của mỗi camera còn lại. D455/D435i và webcam không đồng bộ phần cứng được với nhau."""

    def __init__(self, cfg):
        from .sources import OpenCVSource, RealSenseSource, webcam_options
        fc = cfg["fusion"]
        self.names = [c["name"] for c in fc["cameras"]]
        self.kinds = []
        self.tol = float(fc.get("sync_tol_s", 0.025))
        # Khung của camera phụ cũ hơn mức này so với camera tham chiếu = camera đó đang treo: không dùng khung cũ.
        self.stale_s = float(fc.get("stale_s", max(4 * self.tol, 0.1)))
        # Chờ tối đa bao lâu cho khung khớp của camera phụ; 0 = không chờ, lấy khung gần nhất đang có (không khựng)
        self.pair_wait = float(fc.get("pair_wait_s", self.tol))
        n_rs = sum(str(c.get("source", "realsense")).lower() in REALSENSE_NAMES for c in fc["cameras"])
        self.srcs = []
        try:
            for c in fc["cameras"]:
                src = str(c.get("source", "realsense")).lower()
                if src in REALSENSE_NAMES:
                    if str(c.get("serial") or "").startswith("SERIAL_"):
                        raise SystemExit(f"Camera '{c['name']}': chưa điền serial thật (đang là {c['serial']}).\n"
                                         "Xem serial: python scripts/list_cameras.py")
                    if n_rs > 1 and not c.get("serial"):
                        raise SystemExit("Có nhiều RealSense: đặt 'serial' cho từng camera trong fusion.cameras.\n"
                                         "Xem serial: python scripts/list_cameras.py")
                    rcfg = dict(cfg["camera"].get("realsense", {}), serial=c.get("serial"))
                    self.srcs.append(RealSenseSource({"camera": {"realsense": rcfg}}))
                    self.kinds.append("realsense")
                else:
                    self.srcs.append(OpenCVSource(c["source"], **webcam_options(cfg)))
                    self.kinds.append("opencv")
                    print(f"Camera '{c['name']}' (webcam {c['source']}): {self.srcs[-1].mode}")
        except BaseException:
            self.close()
            raise
        # Độ trễ cố định của từng camera (s): webcam laptop thường trả khung chậm hơn RealSense vài chục ms.
        self.latency = [float(c.get("latency_s", 0.0)) for c in fc["cameras"]]
        self.buf = [deque(maxlen=6) for _ in self.srcs]
        self.stats = [{"frames": 0, "fails": 0, "error": None} for _ in self.srcs]   # để báo lỗi khi mất camera
        self.error = None
        self.cond = threading.Condition()
        self.running = True
        self.last_ref_t = -1.0
        self.threads = [threading.Thread(target=self._loop, args=(i,), daemon=True) for i in range(len(self.srcs))]
        for th in self.threads:
            th.start()

    def _loop(self, i):
        st = self.stats[i]
        while self.running:
            try:
                ok, s = self.srcs[i].read()
            except Exception as e:           # vd RealSense "Frame didn't arrive": thử tiếp, ghi lại để báo
                ok, st["error"] = False, f"{type(e).__name__}: {e}"
            t = time.monotonic() - self.latency[i]
            if not ok:
                st["fails"] += 1
                time.sleep(0.005)
                continue
            st["frames"] += 1
            s.bgr = np.ascontiguousarray(s.bgr).copy()   # không giữ buffer của driver
            with self.cond:
                self.buf[i].append((t, s))
                self.cond.notify_all()

    def read(self, timeout=3.0):
        deadline = time.monotonic() + timeout
        with self.cond:
            while not (self.buf[0] and self.buf[0][-1][0] > self.last_ref_t):
                left = deadline - time.monotonic()
                if left <= 0 or not self.running:
                    self.error = self.describe_stall(timeout)
                    return False, None
                self.cond.wait(left)
            t_ref, ref = self.buf[0][-1]
            self.last_ref_t = t_ref
            views, skew, stale = [ref], 0.0, [False]
            for i in range(1, len(self.buf)):
                wait_until = time.monotonic() + self.pair_wait
                while True:
                    cands = list(self.buf[i])
                    best = min(cands, key=lambda ts: abs(ts[0] - t_ref)) if cands else None
                    newer = best is not None and best[0] >= t_ref - self.tol
                    left = wait_until - time.monotonic()
                    if newer or left <= 0:
                        break
                    self.cond.wait(left)
                if best is None:
                    return False, None
                views.append(best[1])
                lag = abs(best[0] - t_ref)
                stale.append(lag > self.stale_s)
                if lag <= self.stale_s:
                    skew = max(skew, lag)
        return True, MultiSample(views, t_ref, skew, stale)

    def describe_stall(self, timeout):
        """Câu báo lỗi khi camera tham chiếu không gửi khung mới: số khung/lỗi của từng camera."""
        parts = []
        for name, st in zip(self.names, self.stats):
            parts.append(f"  {name}: {st['frames']} khung, {st['fails']} lần đọc lỗi"
                         + (f", lỗi cuối: {st['error']}" if st["error"] else ""))
        return (f"Camera '{self.names[0]}' không gửi khung mới trong {timeout:.0f} s.\n" + "\n".join(parts) +
                "\nKiểm tra: camera có bị app khác chiếm không, cáp/cổng USB, chỉ số webcam (list_cameras.py).")

    def close(self):
        self.running = False
        for th in getattr(self, "threads", []):     # chờ luồng đọc thoát trước khi giải phóng camera
            th.join(timeout=1.0)
        for s in self.srcs:
            try:
                s.close()
            except Exception:
                pass


# ----------------------------------------------------------------------------------------------------------
# Perception nhiều camera
# ----------------------------------------------------------------------------------------------------------
def view_options(cfg):
    """Tuỳ chọn Perception cho từng camera trong fusion.cameras.

    - body_source "front": chỉ camera 0 chạy Pose (các camera khác chỉ Hand); "triangulate": mọi camera chạy Pose.
      fusion.cameras[i].pose ghi đè được (trừ khi triangulate cần Pose ở mọi camera).
    - models.pose_interval / pose_hold_frames: Pose chạy thưa + giữ kết quả khi hụt ngắn.
    - Chỉ điều khiển 1 tay: mỗi camera tìm 1 bàn tay và luôn coi là bàn tay người của tay robot đó.
    """
    fc, mc, mp = cfg["fusion"], cfg["models"], cfg.get("mapping", {})
    front = str(fc.get("body_source", "triangulate")) == "front"
    arms = list(mp.get("robot_arms", ["right", "left"]))
    force = None
    if len(arms) == 1:
        force = arms[0] if mp.get("mode", "direct") == "direct" else ("left" if arms[0] == "right" else "right")
    out = []
    for i, c in enumerate(fc["cameras"]):
        pose = True if not front else bool(c.get("pose", i == 0))
        # Camera có Pose tìm 2 bàn tay để gán theo cổ tay (không nhận nhầm tay kia); camera chỉ Hand tìm 1.
        out.append({"pose_enabled": pose, "pose_interval": int(mc.get("pose_interval", 1)),
                    "pose_hold_frames": int(mc.get("pose_hold_frames", 0)), "force_hand_side": force,
                    "num_hands": 1 if (force and not pose) else 2})
        if not pose and force is None:
            print(f"Cảnh báo: camera '{c.get('name', i)}' chỉ chạy Hand nhưng đang điều khiển 2 tay -> không biết bàn "
                  "tay nào là tay nào, bàn tay ở camera này bị bỏ. Dùng --arms right (hoặc left).")
    if front and not out[0]["pose_enabled"]:
        raise SystemExit("fusion.body_source: front cần camera đầu tiên chạy Pose (fusion.cameras[0].pose: true)")
    return out


class MultiViewPerception:
    """Chạy một Perception (MediaPipe) cho mỗi camera rồi hợp nhất bằng triangulation.

    per_view: danh sách đối tượng có .process(bgr, t) -> Frame (Perception, hoặc bản giả khi test).
    cameras: CameraModel theo cùng thứ tự; camera 0 là khung thế giới.
    """

    def __init__(self, per_view, cameras, fusion_cfg=None, parallel=True):
        fc = fusion_cfg or {}
        self.per_view, self.cams = per_view, cameras
        self.reproj_px = float(fc.get("reproj_thresh_px", 25.0))
        self.depth_w = float(fc.get("depth_weight", 0.3))
        self.depth_tol = float(fc.get("depth_consistency_m", 0.04))
        self.min_vis = float(fc.get("min_visibility", 0.3))
        self.depth_cfg = fc.get("depth", {})
        oc = fc.get("orientation", {})
        self.trackers = {s: HandOrientationTracker(**oc) for s in ("right", "left")}   # (cũ, giữ cho test)
        self.orient = {s: OrientationFusion(**fc.get("orientation_fusion", {})) for s in ("right", "left")}
        # body_source: "triangulate" = vai/khuỷu/cổ tay triangulate từ mọi camera (mọi camera chạy Pose);
        # "front" = lấy từ Pose của camera 0 (điểm world MediaPipe), camera khác chỉ chạy Hand -> nhẹ, mượt.
        self.body_source = str(fc.get("body_source", "triangulate"))
        hc = fc.get("hand", {})
        # Trọng số tin cậy từng camera (fusion.cameras[i].weight, mặc định 1): webcam mờ đặt thấp hơn.
        cams_cfg = fc.get("cameras") or []
        self.view_weight = [float(cams_cfg[i].get("weight", 1.0)) if i < len(cams_cfg) else 1.0
                            for i in range(len(cameras))]
        self.hand_gate_px = float(hc.get("wrist_gate_px", 60.0))
        self.hand_ref_px = float(hc.get("size_ref_px", 35.0))
        self.max_hand_forearm = float(hc.get("max_hand_forearm_deg", 100.0))
        self.wrist_from_hand = float(hc.get("wrist_from_hand", 0.7))
        self.wrist_agree_m = float(hc.get("wrist_agree_m", 0.08))
        self.shapes = {s: HandShape() for s in ("right", "left")}
        self._wrist_blend = {s: 0.0 for s in ("right", "left")}
        self.palms = {s: PalmModel(s, max_rms_m=float(hc.get("palm_max_rms_m", 0.012)))
                      for s in ("right", "left")}
        self._facing = {s: None for s in ("right", "left")}
        self._body_R, self._body_reject = None, 0
        self.view_frames = []
        self.pool = ThreadPoolExecutor(len(per_view)) if parallel and len(per_view) > 1 else None

    @classmethod
    def from_config(cls, cfg, first_sample=None):
        from .perception import Perception
        fc = cfg["fusion"]
        names = [c["name"] for c in fc["cameras"]]
        from .config import ROOT
        calib = Path(fc["calib_file"])
        cams = load_calibration(calib if calib.is_absolute() else ROOT / calib, names)
        if first_sample is not None:
            for cam, s in zip(cams, first_sample.views):
                if s.intrinsics is not None:
                    cam.set_realsense_intrinsics(s.intrinsics)
                check_image_size(cam, s.bgr)
        per_view = [Perception(cfg["models"]["pose"], cfg["models"]["hand"], min_conf=cfg["models"]["min_conf"],
                               orientation_cfg=cfg.get("orientation"), **opts)
                    for opts in view_options(cfg)]
        return cls(per_view, cams, fc)

    def close(self):
        for p in self.per_view:
            if hasattr(p, "close"):
                p.close()
        if self.pool is not None:
            self.pool.shutdown(wait=False)

    # -- quan sát 2D + depth của từng camera ------------------------------------------------------------
    def _view_obs(self, v, sample, fr):
        cam = self.cams[v]
        h, w = sample.bgr.shape[:2]
        o = {"pose": {}, "hand": {}, "size": np.array([w, h], float)}
        depth = sample.depth_m
        intr = sample.intrinsics
        dc = self.depth_cfg

        def lift(u, vv, radius):
            if depth is None or intr is None:
                return None
            z = sample_depth(depth, u, vv, radius=radius, min_m=dc.get("min_depth_m", 0.3),
                             max_m=dc.get("max_depth_m", 6.0), max_delta_m=dc.get("max_local_delta_m", 0.15))
            if z is None:
                return None
            xy = cam.normalize([[u, vv]])[0]
            return cam.to_world(np.array([xy[0] * z, xy[1] * z, z]))

        cw = self.view_weight[v]
        if fr.pose_2d is not None:
            P = fr.pose_2d
            px = P[:, :2] * [w, h]
            norm = cam.normalize(px[list(BODY_IDS)])
            for k, i in enumerate(BODY_IDS):
                vis = float(P[i, 2])
                if vis >= self.min_vis:
                    o["pose"][i] = (norm[k], vis, lift(px[i, 0], px[i, 1], dc.get("patch_radius", 3)), px[i])
        for h2, side in fr.hands_2d:
            if side not in ("right", "left") or side in o["hand"]:
                continue
            px = np.asarray(h2) * [w, h]
            conf = float(fr.arms[side].conf.get("hand", 0.0)) if side in fr.arms else 0.0
            if conf <= 0.0:
                conf = 0.5
            # Bàn tay nhỏ trên ảnh (xa, độ phân giải thấp) -> điểm kém chính xác -> trọng số thấp hơn
            size = np.linalg.norm(px[H_MIDDLE_MCP] - px[H_WRIST]) * REF_FX / max(cam.fx, 1e-6)
            trust = cw * float(np.clip(size / self.hand_ref_px, 0.4, 1.0))    # chỉ là trọng số tương đối
            norm = cam.normalize(px)
            lifted = [lift(px[j, 0], px[j, 1], dc.get("hand_patch_radius", 2)) for j in range(21)]
            o["hand"][side] = (norm, conf, lifted, px, trust)
        return o

    # -- hợp nhất ----------------------------------------------------------------------------------------
    def fuse(self, views_obs, t):
        if self.body_source == "front":
            return self._fuse_front(views_obs, t)
        info = {"views": len(self.cams), "points": {}}
        W = np.full((33, 3), np.nan)
        vis = np.zeros(33)
        for i in BODY_IDS:
            obs = [(self.cams[v], o["pose"][i][0], o["pose"][i][1], self.view_weight[v])
                   for v, o in enumerate(views_obs) if i in o["pose"]]
            dep = [(self.cams[v], o["pose"][i][2], o["pose"][i][1]) for v, o in enumerate(views_obs)
                   if i in o["pose"] and o["pose"][i][2] is not None]
            X, c, pi = fuse_point(obs, dep, self.reproj_px, self.depth_w, self.depth_tol)
            info["points"][i] = pi
            if X is not None:
                W[i], vis[i] = X, c
        arms = {"right": ArmObs(), "left": ArmObs()}
        if not (np.all(np.isfinite(W[[L_SH, R_SH]])) and min(vis[L_SH], vis[R_SH]) > 0):
            self._body_R = None                          # mất người: lần sau nhận khung thân mới
            for s in self.orient:
                self.orient[s].update(None)
            return Frame(arms, None, [], None, t, fusion=info), W
        if not np.all(np.isfinite(W[[L_HIP, R_HIP]])):
            vis[L_HIP] = vis[R_HIP] = 0.0
        R_body, origin = body_frame(W, vis)
        if self._body_R is None or self._body_reject >= 10:
            self._body_R, self._body_reject = R_body, 0
        elif np.rad2deg(rotation_distance(self._body_R, R_body)) <= 45:
            self._body_R, self._body_reject = slerp_rotation(self._body_R, R_body, 0.35), 0
        else:
            self._body_reject += 1                       # nhảy lớn: giữ khung cũ, 10 khung liền thì nhận khung mới
        Rb = self._body_R
        depth_used = {}
        for side, (i_s, i_e, i_w) in ARM_IDX.items():
            ob = arms[side]
            self._gate_hands(side, W[i_w], views_obs, info)
            hand_root = self._fuse_hand(side, ob, views_obs, Rb, info, W[i_e], W[i_w])
            if np.all(np.isfinite(W[[i_s, i_e, i_w]])):
                # Đồng bộ tay-bàn tay: cổ tay của cẳng tay (J3/J4) và gốc khung bàn tay (J5-J7) dùng CHUNG một
                # điểm. Cổ tay của MediaPipe Hand chính xác hơn cổ tay Pose; chỉ dùng khi hai điểm gần nhau. Tỉ lệ
                # trộn đổi dần (0,15/khung) để bàn tay lúc có lúc mất không làm cổ tay (và J3/J4) giật.
                agree = hand_root is not None and np.linalg.norm(hand_root - W[i_w]) < self.wrist_agree_m
                goal = self.wrist_from_hand if agree else 0.0
                a = self._wrist_blend[side]
                a = self._wrist_blend[side] = a + float(np.clip(goal - a, -0.15, 0.15))
                if agree and a > 0:
                    W[i_w] = (1 - a) * W[i_w] + a * hand_root
                    info["points"][i_w]["hand_root"] = True
                ob.s, ob.e, ob.w = (Rb.T @ (W[k] - origin) for k in (i_s, i_e, i_w))
                ob.conf["upper"] = float(min(vis[i_s], vis[i_e]))
                ob.conf["fore"] = float(min(vis[i_e], vis[i_w]))
            depth_used[side] = sum(info["points"][k]["depth"] > 0 for k in (i_s, i_e, i_w))
        return Frame(arms, None, [], Rb, t, depth_used=depth_used, body_origin=origin, fusion=info), W

    def _fuse_front(self, views_obs, t):
        """Vai/khuỷu/cổ tay + khung thân từ Pose camera 0 (điểm world MediaPipe, trục cùng hướng camera 0);
        bàn tay hợp nhất từ mọi camera. Giống cách bản Openarm_Teleop chạy mượt trên robot thật."""
        info = {"views": len(self.cams), "points": {}, "body": "front"}
        W = np.full((33, 3), np.nan)
        f0 = self.view_frames[0] if self.view_frames else None
        arms = {"right": ArmObs(), "left": ArmObs()}
        if f0 is None or f0.body_R is None:
            for s in self.orient:
                self.orient[s].update(None)
                arms[s].hand_orientation_mode = "NONE"
            return Frame(arms, None, [], None, t, fusion=info), W
        Rb = f0.body_R
        for side in ("right", "left"):
            src, ob = f0.arms[side], arms[side]
            ob.s, ob.e, ob.w = src.s, src.e, src.w
            ob.conf["upper"], ob.conf["fore"] = src.conf["upper"], src.conf["fore"]
            fore = Rb @ (src.w - src.e) if src.w is not None and src.e is not None else None
            self._gate_front(side, f0, views_obs, info)
            self._fuse_hand(side, ob, views_obs, Rb, info, np.zeros(3) if fore is not None else None, fore)
            if ob.grip is None and src.grip is not None:
                ob.grip = src.grip
        return Frame(arms, None, [], Rb, t, depth_used={}, body_origin=f0.body_origin, fusion=info), W

    def _gate_front(self, side, f0, views_obs, info):
        """Chế độ front: bàn tay ở camera phụ (chỉ Hand, tự gán) phải khớp cổ tay Pose của camera 0:
        - camera 0 cũng thấy bàn tay: triangulate cổ tay bàn tay từ 2 camera, sai số chiếu lại lớn -> camera phụ
          đang thấy tay khác -> bỏ;
        - camera 0 không thấy bàn tay: cổ tay từ depth của camera phụ chiếu vào camera 0 phải gần cổ tay Pose;
          không có depth thì không kiểm chứng được -> bỏ."""
        rejected = 0
        cam0 = self.cams[0]
        size0 = views_obs[0].get("size")
        wr0 = None
        if f0.pose_2d is not None and size0 is not None:
            wr0 = f0.pose_2d[ARM_IDX[side][2], :2] * size0
        h0 = views_obs[0]["hand"].get(side)
        for v in range(1, len(views_obs)):
            hv = views_obs[v]["hand"].get(side)
            if hv is None:
                continue
            ok = False
            if h0 is not None:
                X = triangulate_weighted([(cam0, h0[0][H_WRIST], 1.0), (self.cams[v], hv[0][H_WRIST], 1.0)])
                ok = X is not None and max(reprojection_px(cam0, X, h0[0][H_WRIST]),
                                           reprojection_px(self.cams[v], X, hv[0][H_WRIST])) <= self.hand_gate_px
            elif wr0 is not None and hv[2][H_WRIST] is not None:
                p0 = cam0.project(np.asarray(hv[2][H_WRIST])[None])[0]
                ok = bool(np.all(np.isfinite(p0))) and \
                    np.linalg.norm(p0 - wr0) * REF_FX / max(cam0.fx, 1e-6) <= self.hand_gate_px
            if not ok:
                del views_obs[v]["hand"][side]
                rejected += 1
        info[f"hand_{side}_rejected"] = rejected

    def _orientation_extras(self, side, views_obs, seen):
        """Nguồn phụ cho hướng bàn tay (khung world): MediaPipe world từng camera, lòng bàn tay từ depth."""
        extras = []
        for v in seen:
            cam = self.cams[v]
            f = self.view_frames[v] if v < len(self.view_frames) else None
            R_cam = getattr(f.arms[side], "hand_R_cam", None) if f is not None else None
            if R_cam is not None:
                extras.append((cam.R.T @ R_cam, 0.5, f"rgb:{cam.name}"))
            lifted = views_obs[v]["hand"][side][2]
            P = np.full((21, 3), np.nan)
            for j in PALM_IDS:
                if lifted[j] is not None:
                    P[j] = lifted[j]
            Rd, _ = palm_frame_from_depth(P, side=side)
            if Rd is not None:
                extras.append((Rd, 0.6, f"depth:{cam.name}"))
        return extras

    def _gate_hands(self, side, wrist_w, views_obs, info):
        """Bỏ bàn tay ở camera nào mà cổ tay của nó xa cổ tay Pose (đã triangulate) chiếu vào camera đó."""
        rejected = 0
        if not np.all(np.isfinite(wrist_w)):
            info[f"hand_{side}_rejected"] = 0
            return
        for v, o in enumerate(views_obs):
            if side not in o["hand"]:
                continue
            cam = self.cams[v]
            pw = cam.project(np.asarray(wrist_w)[None])[0]
            d = np.linalg.norm(o["hand"][side][3][H_WRIST] - pw) * REF_FX / max(cam.fx, 1e-6)
            if not np.isfinite(d) or d > self.hand_gate_px:
                del o["hand"][side]
                rejected += 1
        info[f"hand_{side}_rejected"] = rejected

    def _fuse_hand(self, side, ob, views_obs, Rb, info, elbow_w=None, wrist_w=None):
        """Triangulate 21 điểm, lọc đốt bất thường, hướng lòng bàn tay (Kabsch), kiểm tra với cẳng tay.
        Trả vị trí cổ tay của bàn tay (world) hoặc None."""
        seen = [v for v, o in enumerate(views_obs) if side in o["hand"]]
        if not seen:
            ob.hand_orientation_mode = self.orient[side].update(None)[1]
            info[f"hand_{side}"] = {"views": 0, "quality": 0.0, "mode": ob.hand_orientation_mode, "points": 0,
                                    "rejected": info.get(f"hand_{side}_rejected", 0)}
            return None
        facing = self._facing[side]
        pts = np.full((21, 3), np.nan)
        confs = np.zeros(21)
        for j in range(21):
            obs, dep = [], []
            for v in seen:
                norm, conf, lifted, _, trust = views_obs[v]["hand"][side]
                tv = trust * (1.0 if facing is None else 0.25 + 0.75 * facing[v])
                obs.append((self.cams[v], norm[j], conf, tv))
                if lifted[j] is not None:
                    dep.append((self.cams[v], lifted[j], conf * tv))
            X, c, _ = fuse_point(obs, dep, self.reproj_px, self.depth_w, self.depth_tol)
            if X is not None:
                pts[j], confs[j] = X, c
        pts, dropped = self.shapes[side].filter(pts)
        confs[~np.all(np.isfinite(pts), axis=1)] = 0.0
        # Số ngón xoè: từ điểm 3D; thiếu điểm (chỉ 1 camera thấy, không depth) thì lấy từ điểm world của từng camera
        ob.hand_open_fingers = max([open_finger_count(pts)] +
                                   [self.view_frames[v].arms[side].hand_open_fingers for v in seen
                                    if v < len(self.view_frames)])
        if np.all(np.isfinite(pts[[H_WRIST, H_MIDDLE_MCP, H_THUMB_TIP, H_INDEX_TIP]])):
            ob.grip = float(np.linalg.norm(pts[H_THUMB_TIP] - pts[H_INDEX_TIP]) /
                            max(np.linalg.norm(pts[H_MIDDLE_MCP] - pts[H_WRIST]), 1e-6))
        prev_q = max((self._facing[side][v] for v in seen), default=0.0) if self._facing[side] else 0.0
        raw_R, center, fit_rms, fit_mode = self.palms[side].estimate(pts, confs[list(PALM_IDS)], prev_q)
        quality, face_v = 0.0, [0.0] * len(self.cams)
        anat = hand_forearm_angle(raw_R, elbow_w, wrist_w)
        if raw_R is not None and np.isfinite(anat) and anat > self.max_hand_forearm:
            raw_R, fit_mode = None, "ANAT"          # bàn tay gập quá mức cổ tay người làm được: quan sát sai
        if raw_R is not None:
            for v, cam in enumerate(self.cams):
                face_v[v] = abs(float(raw_R[:, 2] @ unit(center - cam.center)))
            quality = max(face_v[v] for v in seen)
            self._facing[side] = face_v
        extras = [(Re, c, n) for Re, c, n in self._orientation_extras(side, views_obs, seen)
                  if not (hand_forearm_angle(Re, elbow_w, wrist_w) > self.max_hand_forearm)]
        # Độ tin cậy của nguồn 3D: trung bình các điểm lòng bàn tay (đã gồm phạt khi hai camera mâu thuẫn), giảm
        # khi chỉ còn 3 điểm; trust của camera chỉ dùng để trộn, không hạ độ tin cậy.
        pc = confs[list(PALM_IDS)]
        n_ok = int(np.count_nonzero(pc > 0))
        palm_conf = float(np.mean(pc[pc > 0]) * min(1.0, n_ok / 4)) if raw_R is not None and n_ok else 0.0
        R, mode, o_conf, used = self.orient[side].update(raw_R, max(palm_conf, 0.05), extras,
                                                         base_strong=raw_R is not None and len(seen) >= 2)
        ob.hand_orientation_mode = mode if (raw_R is not None or mode not in ("HOLD",)) else f"HOLD {fit_mode}"
        ob.conf["hand"] = min(o_conf, palm_conf) if (used and used[0] == "3d") else o_conf
        if R is not None:
            ob.H = Rb.T @ R
            ob.hand_R_cam = R
            ob.hand_center_cam = center
            if center is not None:
                axes = np.array([center] + [center + R[:, k] * 0.08 for k in range(3)])
                px = self.cams[0].project(axes)
                if np.all(np.isfinite(px)):
                    ob.hand_axes_px = px
        info[f"hand_{side}"] = {"views": len(seen), "quality": quality, "mode": ob.hand_orientation_mode,
                                "points": int(np.count_nonzero(np.all(np.isfinite(pts), axis=1))),
                                "fit": fit_mode, "fit_mm": 1000 * fit_rms if np.isfinite(fit_rms) else np.nan,
                                "bones_dropped": dropped, "forearm_deg": anat,
                                "rejected": info.get(f"hand_{side}_rejected", 0), "sources": used}
        root = pts[H_WRIST]
        return root if np.all(np.isfinite(root)) and confs[H_WRIST] > 0 else None

    def process(self, msample: MultiSample) -> Frame:
        for cam, s in zip(self.cams, msample.views):
            if cam.rs_intr is None and s.intrinsics is not None:
                cam.set_realsense_intrinsics(s.intrinsics)
        stale = msample.stale or [False] * len(msample.views)
        jobs = [(p, s.bgr) for p, s, st in zip(self.per_view, msample.views, stale) if not st]
        if self.pool is not None:
            done = list(self.pool.map(lambda a: a[0].process(a[1], msample.t), jobs))
        else:
            done = [p.process(b, msample.t) for p, b in jobs]
        it = iter(done)
        # Camera stale: không chạy MediaPipe, coi như không thấy gì (không trộn khung cũ vào triangulation)
        frames = [Frame({"right": ArmObs(), "left": ArmObs()}, None, [], None, msample.t) if st else next(it)
                  for st in stale]
        self.view_frames = frames
        views_obs = [self._view_obs(v, s, f) for v, (s, f) in enumerate(zip(msample.views, frames))]
        fr, W = self.fuse(views_obs, msample.t)
        fr.pose_2d, fr.hands_2d = frames[0].pose_2d, frames[0].hands_2d
        fr.fusion["skew_ms"] = 1000.0 * msample.skew_s
        fr.fusion["stale"] = [self.cams[v].name for v, st in enumerate(stale) if st]
        self.last_world = W
        return fr

    def draw(self, msample, fused: Frame, height=360):
        """Ảnh từng camera (khung xương MediaPipe) + điểm hợp nhất chiếu lại (tím) để thấy hai camera có khớp."""
        from .viz import draw_human, put_lines
        tiles = []
        W = getattr(self, "last_world", None)
        for v, (s, f) in enumerate(zip(msample.views, self.view_frames)):
            img = draw_human(s.bgr.copy(), f if v else fused)
            if W is not None:
                ids = [i for i in (L_SH, R_SH, L_EL, R_EL, L_WR, R_WR) if np.all(np.isfinite(W[i]))]
                if ids:
                    for p in self.cams[v].project(W[ids]):
                        if np.all(np.isfinite(p)):
                            cv2.circle(img, (int(p[0]), int(p[1])), 7, (255, 0, 255), 2)
            put_lines(img, [f"{self.cams[v].name}"], org=(10, img.shape[0] - 14))
            scale = height / img.shape[0]
            tiles.append(cv2.resize(img, (int(img.shape[1] * scale), height)))
        return np.hstack(tiles)
