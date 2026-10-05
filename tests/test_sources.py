"""OpenCVSource: webcam chỉ nhận MJPG khi đặt SAU kích thước (như một số bản OpenCV/driver) vẫn được cấu hình đúng."""
import cv2
import numpy as np

import openarm_shadow.cameras.sources as sources

MJPG = cv2.VideoWriter_fourcc(*"MJPG")
YUYV = cv2.VideoWriter_fourcc(*"YUYV")


class FakeCap:
    """Webcam giả: YUYV chỉ tới 640x480; 1280x720 cần MJPG; MJPG chỉ nhận khi kích thước đã là 1280x720;
    đặt fps quay về 640x480 YUYV (driver reset)."""

    def __init__(self, *a):
        self.w, self.h, self.cc, self.fps = 640, 480, YUYV, 30.0

    def isOpened(self):
        return True

    def set(self, prop, val):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            self.w = int(val)
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            self.h = int(val)
        elif prop == cv2.CAP_PROP_FOURCC:
            if int(val) == MJPG and self.w != 1280:
                return False
            self.cc = int(val)
        elif prop == cv2.CAP_PROP_FPS:
            self.w, self.h, self.cc = 640, 480, YUYV
        return True

    def _mode(self):
        if self.cc == YUYV and self.w > 640:
            return 640, 480
        return self.w, self.h

    def get(self, prop):
        w, h = self._mode()
        return {cv2.CAP_PROP_FRAME_WIDTH: w, cv2.CAP_PROP_FRAME_HEIGHT: h,
                cv2.CAP_PROP_FOURCC: self.cc, cv2.CAP_PROP_FPS: self.fps}.get(prop, 0)

    def read(self):
        w, h = self._mode()
        return True, np.zeros((h, w, 3), np.uint8)

    def release(self):
        pass


def test_webcam_mode_retries_until_it_sticks(monkeypatch):
    monkeypatch.setattr(sources.cv2, "VideoCapture", FakeCap)
    src = sources.OpenCVSource(0, 1280, 720, fourcc="MJPG", fps=30)
    assert src.warning is None
    assert src.mode.startswith("1280x720 MJPG")
    ok, s = src.read()
    assert s.bgr.shape[:2] == (720, 1280)


def test_webcam_mode_warns_when_unsupported(monkeypatch):
    monkeypatch.setattr(sources.cv2, "VideoCapture", FakeCap)
    src = sources.OpenCVSource(0, 1920, 1080, fourcc="MJPG", fps=30)
    assert src.warning and "1920x1080" in src.warning and "list-formats-ext" in src.warning
