"""Phát lại dữ liệu thô ghi bằng scripts/record_multicam_raw.py như một nguồn nhiều camera.

Mỗi camera trong thư mục ghi: rgb.avi, timestamps.csv (frame, host_arrival_s, device_timestamp_s) và với RealSense
thêm depth_u16_mm.raw. RawReplaySource.read() trả MultiSample giống MultiCameraSource (sync "latest", không chờ):
mỗi khung camera tham chiếu (camera đầu) ghép với khung có thời điểm gần nhất của từng camera còn lại, lệch hơn
max_skew_s thì đánh dấu stale. Thời điểm = host_arrival_s − latency_s như lúc chạy thật.

Nội tham số RealSense: đọc từ intrinsics.yaml trong thư mục ghi (record_multicam_raw.py lưu từ nay). Bản ghi cũ không
có file này thì truyền một file tương tự (intrinsics=...): {tên camera: {fx, fy, ppx, ppy}}.
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import yaml

from .multicam import MultiSample
from .sources import CameraSample


def load_intrinsics(path):
    """{tên camera: dict fx, fy, ppx, ppy (+ width, height, model, coeffs)} từ file YAML."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    return dict(data.get("cameras", data))


class _CameraTrack:
    """Một camera đã ghi: đọc tuần tự, giữ khung hiện tại và khung kế để chọn khung gần thời điểm nhất."""

    def __init__(self, cam_dir, meta, latency_s, intrinsics):
        self.name = cam_dir.name
        with (cam_dir / "timestamps.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.t = np.array([float(r["host_arrival_s"]) for r in rows]) - float(latency_s)
        dev = [r.get("device_timestamp_s") or "" for r in rows]
        self.dev_t = [float(v) if v else None for v in dev]
        self.cap = cv2.VideoCapture(str(cam_dir / "rgb.avi"))
        if not self.cap.isOpened():
            raise SystemExit(f"Không mở được {cam_dir / 'rgb.avi'}")
        self.depth = None
        depth_file = cam_dir / "depth_u16_mm.raw"
        if depth_file.is_file():
            h, w = int(meta["depth_height"]), int(meta["depth_width"])
            self.depth = np.memmap(depth_file, dtype="<u2", mode="r").reshape(-1, h, w)
            if intrinsics is None:
                raise SystemExit(f"Camera '{self.name}' có depth nhưng thiếu nội tham số (intrinsics.yaml).")
        self.intrinsics = intrinsics
        n = len(self.t) if self.depth is None else min(len(self.t), len(self.depth))
        self.n = min(n, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or n)
        self.k, self.bgr = -1, None

    def _advance(self):
        ok, img = self.cap.read()
        if not ok:
            return False
        self.k, self.bgr = self.k + 1, img
        return True

    def seek_nearest(self, t):
        """Đọc tới khung có thời điểm gần t nhất (chỉ đi tới, t tăng dần). -> chỉ số khung hoặc None."""
        if self.k < 0 and not self._advance():
            return None
        while self.k + 1 < self.n and abs(self.t[self.k + 1] - t) <= abs(self.t[self.k] - t):
            if not self._advance():
                break
        return self.k

    def sample(self):
        depth_m = None if self.depth is None else self.depth[self.k].astype(np.float32) * 1e-3
        return CameraSample(self.bgr.copy(), depth_m, self.intrinsics, self.dev_t[self.k])

    def close(self):
        self.cap.release()


class RawReplaySource:
    """Nguồn nhiều camera từ thư mục ghi thô. names: thứ tự camera (camera đầu = tham chiếu).

    start_s: bỏ qua chừng này giây đầu; max_frames: số khung tham chiếu tối đa; stride: chỉ dùng 1/stride khung
    tham chiếu (mô phỏng perception chậm hơn camera)."""

    def __init__(self, root, names, latency_s=None, max_skew_s=0.04, intrinsics=None, start_s=0.0,
                 max_frames=None, stride=1):
        root = Path(root)
        meta = yaml.safe_load((root / "metadata.yaml").read_text())
        if intrinsics is None and (root / "intrinsics.yaml").is_file():
            intrinsics = load_intrinsics(root / "intrinsics.yaml")
        intrinsics = intrinsics or {}
        latency_s = latency_s or [0.0] * len(names)
        self.names = list(names)
        self.tracks = [_CameraTrack(root / n, meta["cameras"][n], lat, intrinsics.get(n))
                       for n, lat in zip(self.names, latency_s)]
        self.stale_s = float(max_skew_s)
        self.stride = max(1, int(stride))
        ref = self.tracks[0]
        self.t0 = ref.t[0] + float(start_s)
        self.max_frames = max_frames
        self.count = 0
        self.error = None

    def read(self, timeout=None):
        ref = self.tracks[0]
        if self.max_frames is not None and self.count >= self.max_frames:
            self.error = "hết số khung yêu cầu"
            return False, None
        # Khung tham chiếu kế tiếp: bỏ (stride - 1) khung, bỏ phần trước start_s
        while True:
            if ref.k + 1 >= ref.n or not ref._advance():
                self.error = "hết dữ liệu"
                return False, None
            if ref.t[ref.k] >= self.t0 and (ref.k % self.stride == 0):
                break
        t_ref = float(ref.t[ref.k])
        views, skew, stale, lags = [ref.sample()], 0.0, [False], [0.0]
        for tr in self.tracks[1:]:
            k = tr.seek_nearest(t_ref)
            if k is None:
                self.error = f"camera '{tr.name}' không có khung"
                return False, None
            lag = abs(float(tr.t[k]) - t_ref)
            views.append(tr.sample())
            lags.append(lag)
            stale.append(lag > self.stale_s)
            if lag <= self.stale_s:
                skew = max(skew, lag)
        self.count += 1
        return True, MultiSample(views, t_ref, skew, stale, lags)

    def close(self):
        for tr in self.tracks:
            tr.close()
