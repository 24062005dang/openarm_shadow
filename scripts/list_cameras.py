#!/usr/bin/env python3
"""Liệt kê camera: RealSense (tên, serial, USB) và webcam OpenCV mở được (chỉ số 0..5).

    python scripts/list_cameras.py
Điền serial vào fusion.cameras trong config (vd config/fusion_2cam.yaml).
"""
import cv2


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
    print("Webcam OpenCV (RealSense cũng có thể hiện ở đây, bỏ qua các chỉ số đó):")
    for i in range(6):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, img = cap.read()
            if ok:
                print(f"  chỉ số {i}: {img.shape[1]}x{img.shape[0]}")
        cap.release()


if __name__ == "__main__":
    main()
