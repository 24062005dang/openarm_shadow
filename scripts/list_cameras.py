#!/usr/bin/env python3
"""Liệt kê camera: RealSense (tên, serial, USB) và webcam OpenCV mở được (chỉ số, tên thiết bị).

    python scripts/list_cameras.py
Điền chỉ số webcam laptop và serial D455 / D435i vào fusion.cameras (config/fusion_3cam.yaml).
"""
import os
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")    # thử mở chỉ số không có camera thì OpenCV in cảnh báo
import cv2  # noqa: E402


def v4l2_name(i):
    """Tên thiết bị /dev/video<i> trên Linux (vd 'Integrated_Webcam_HD', 'Intel(R) RealSense(TM) Depth Ca...')."""
    f = Path(f"/sys/class/video4linux/video{i}/name")
    return f.read_text().strip() if f.is_file() else ""


def main():
    try:
        import pyrealsense2 as rs
        devs = list(rs.context().devices)
        print(f"RealSense: {len(devs)} thiết bị")
        for d in devs:
            info = lambda k: d.get_info(k) if d.supports(k) else "?"
            usb = info(rs.camera_info.usb_type_descriptor)
            print(f"  {info(rs.camera_info.name):28s} serial {info(rs.camera_info.serial_number)}  USB {usb}"
                  + ("   <-- USB 2: đổi cổng/cáp USB 3" if str(usb).startswith("2") else ""))
    except ImportError:
        print("Chưa cài pyrealsense2 (pip install pyrealsense2)")
    print("Webcam OpenCV (chỉ số dùng cho 'source:'; các dòng RealSense thì bỏ qua):")
    linux = Path("/sys/class/video4linux").is_dir()
    for i in range(10):
        name = v4l2_name(i)
        if "realsense" in name.lower() or (linux and not Path(f"/dev/video{i}").exists()):
            continue
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, img = cap.read()
            if ok:
                print(f"  chỉ số {i}: {img.shape[1]}x{img.shape[0]}  {name}")
        cap.release()


if __name__ == "__main__":
    main()
