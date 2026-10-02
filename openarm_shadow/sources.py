"""Nguồn ảnh cho teleop: OpenCV thông thường hoặc Intel RealSense RGB-D."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class CameraSample:
    bgr: np.ndarray
    depth_m: np.ndarray | None = None
    intrinsics: object | None = None
    timestamp_s: float | None = None


def set_v4l2_controls(device, controls):
    """Đặt control của webcam qua v4l2-ctl (Linux, gói v4l-utils): vd {"sharpness": 5, "auto_exposure": 1}.
    Tên control xem bằng: v4l2-ctl -d /dev/video0 --list-ctrls. Trả list lỗi (rỗng nếu ổn)."""
    import shutil
    import subprocess
    if not controls:
        return []
    if shutil.which("v4l2-ctl") is None:
        return ["chưa có v4l2-ctl (sudo apt install v4l-utils): bỏ qua camera.v4l2"]
    errs = []
    for name, value in controls.items():      # từng control một: control sai tên không làm hỏng các control khác
        r = subprocess.run(["v4l2-ctl", "-d", str(device), f"--set-ctrl={name}={value}"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            errs.append(f"{name}={value}: {(r.stderr or r.stdout).strip()}")
    return errs


class OpenCVSource:
    """Webcam / video / luồng mạng. Webcam: fourcc="MJPG" để được 1280x720 @ 30 fps (định dạng mặc định YUYV
    của nhiều webcam laptop chỉ ra ~10 fps ở 720p); v4l2 = control của webcam (Linux)."""

    def __init__(self, source, width, height, fourcc=None, fps=None, v4l2=None):
        is_index = str(source).isdigit()
        if is_index and Path("/dev").joinpath(f"video{int(source)}").exists():
            self.cap = cv2.VideoCapture(int(source), cv2.CAP_V4L2)
        else:
            self.cap = cv2.VideoCapture(int(source) if is_index else source)
        if not self.cap.isOpened():
            raise SystemExit(f"Không mở được nguồn video: {source}")
        if is_index:
            # Bớt khung cũ đọng trong buffer driver (giảm trễ). Không đặt 1: driver không có buffer trống để ghi
            # trong lúc khung đang được đọc -> mất 1/2 khung (webcam Sonix 720p MJPG: 15 thay vì 30 fps).
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.warning = None
        if is_index:
            self._configure_webcam(int(source), width, height, fourcc, fps)
        else:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if is_index and v4l2:
            for e in set_v4l2_controls(f"/dev/video{int(source)}", v4l2):
                print(f"Cảnh báo webcam {source}: {e}")
        self.mode = self.describe()

    def _fourcc(self):
        cc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        return "".join(chr((cc >> 8 * i) & 0xFF) for i in range(4)) if cc else "?"

    def _configure_webcam(self, index, width, height, fourcc, fps):
        """Đặt định dạng/kích thước/fps rồi KIỂM TRA bằng một khung thật. Tuỳ bản OpenCV/driver, định dạng phải đặt
        trước hoặc sau kích thước, và đặt fps có thể làm driver quay về chế độ mặc định: thử lần lượt các thứ tự."""
        cc = cv2.VideoWriter_fourcc(*str(fourcc)) if fourcc else None
        W, H, F, P = ("w", width), ("h", height), ("fourcc", cc), ("fps", fps)
        orders = [[F, W, H, P], [W, H, F, P], [F, W, H], [W, H, F]]
        props = {"w": cv2.CAP_PROP_FRAME_WIDTH, "h": cv2.CAP_PROP_FRAME_HEIGHT, "fourcc": cv2.CAP_PROP_FOURCC,
                 "fps": cv2.CAP_PROP_FPS}
        for order in orders:
            for key, val in order:
                if val:
                    self.cap.set(props[key], float(val) if key == "fps" else val)
            ok, img = self.cap.read()             # nhiều driver chỉ áp dụng chế độ mới khi đọc khung đầu
            size_ok = ok and img.shape[1] == int(width) and img.shape[0] == int(height)
            fmt_ok = not fourcc or self._fourcc().upper() == str(fourcc).upper()
            if size_ok and fmt_ok:
                return
        got = self.describe()
        self.warning = (f"webcam {index} không chạy được {width}x{height}"
                        + (f" {fourcc}" if fourcc else "") + f", đang chạy {got}. Xem chế độ webcam hỗ trợ: "
                        f"v4l2-ctl -d /dev/video{index} --list-formats-ext (hoặc scripts/webcam_check.py)")
        print("Cảnh báo:", self.warning)

    def describe(self):
        return (f"{int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
                f"{self._fourcc()} {self.cap.get(cv2.CAP_PROP_FPS):.0f} fps")

    def read(self):
        ok, bgr = self.cap.read()
        return ok, CameraSample(bgr) if ok else None

    def close(self):
        self.cap.release()


class RealSenseSource:
    """D455: RGB và depth đã lọc, đồng bộ, align theo pixel của RGB."""

    def __init__(self, cfg):
        try:
            import pyrealsense2 as rs
        except ImportError as e:
            raise SystemExit("Thiếu pyrealsense2. Cài: pip install pyrealsense2") from e
        self.rs = rs
        self.pipe = rs.pipeline()
        rcfg = cfg["camera"].get("realsense", {})
        width = int(rcfg.get("width", 640))
        height = int(rcfg.get("height", 480))
        fps = int(rcfg.get("fps", 30))
        serial = rcfg.get("serial")
        rsc = rs.config()
        if serial:
            rsc.enable_device(str(serial))
        rsc.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        rsc.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        try:
            profile = self.pipe.start(rsc)
        except RuntimeError as e:
            raise SystemExit(f"Không mở được RealSense RGB-D {width}x{height}@{fps}: {e}") from e
        self.align = rs.align(rs.stream.color)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.depth_to_disparity = rs.disparity_transform(True)
        self.disparity_to_depth = rs.disparity_transform(False)
        self.spatial = rs.spatial_filter() if rcfg.get("spatial_filter", True) else None
        self.temporal = rs.temporal_filter() if rcfg.get("temporal_filter", True) else None

    def read(self):
        try:
            frames = self.align.process(self.pipe.wait_for_frames(5000))
        except RuntimeError:
            return False, None
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return False, None
        # Lọc trong disparity domain để bộ lọc spatial xử lý stereo depth đồng đều hơn.
        # Không hole-fill mạnh: ở biên bàn tay nó dễ kéo depth của nền vào giữa các ngón.
        depth_frame = self.depth_to_disparity.process(depth_frame)
        if self.spatial is not None:
            depth_frame = self.spatial.process(depth_frame)
        if self.temporal is not None:
            depth_frame = self.temporal.process(depth_frame)
        depth_frame = self.disparity_to_depth.process(depth_frame)
        bgr = np.asanyarray(color_frame.get_data())
        depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale
        # Sau rs.align(depth -> color), profile của depth đã nằm trong pixel space của RGB.
        intr = depth_frame.profile.as_video_stream_profile().intrinsics
        timestamp_s = float(color_frame.get_timestamp()) * 1e-3
        return True, CameraSample(bgr, depth_m, intr, timestamp_s)

    def close(self):
        self.pipe.stop()


def open_source(source, cfg):
    required = str(cfg.get("camera", {}).get("required_source", "")).lower()
    is_realsense = str(source).lower() in {"realsense", "rs", "d455"}
    if required in {"realsense", "rs", "d455"} and not is_realsense:
        raise SystemExit("Cấu hình này yêu cầu Intel RealSense D455; hãy chạy với --source realsense")
    if is_realsense:
        return RealSenseSource(cfg)
    return OpenCVSource(source, **webcam_options(cfg))


def webcam_options(cfg):
    """Tham số webcam từ cfg["camera"]: width, height, fourcc, fps, v4l2."""
    c = cfg["camera"]
    return {"width": c["width"], "height": c["height"], "fourcc": c.get("fourcc"), "fps": c.get("fps"),
            "v4l2": c.get("v4l2") or {}}
