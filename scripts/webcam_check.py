#!/usr/bin/env python3
"""Kiểm tra độ nét webcam: chế độ thật đang chạy (độ phân giải, định dạng, fps) + điểm nét, để so các cách chỉnh.

    python scripts/webcam_check.py --config config/fusion_2cam.yaml          # webcam = camera 'front'
    python scripts/webcam_check.py --source 0 --width 1280 --height 720 --fourcc MJPG

Điểm nét = phương sai Laplacian ở giữa ảnh (cao hơn = nét hơn). Chỉ so sánh được khi cùng cảnh, cùng khoảng
cách: đặt tờ bảng ChArUco (hoặc trang chữ) cách webcam ~1,3 m, đổi một thứ mỗi lần rồi xem điểm tăng hay giảm.
In thêm danh sách định dạng và control của webcam nếu có v4l2-ctl (sudo apt install v4l-utils).
Phím: s = lưu ảnh webcam_check_<thời gian>.png, q/Esc = thoát.
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openarm_shadow.config import load_config
from openarm_shadow.camera.sources import OpenCVSource, webcam_options
from openarm_shadow.display.viz import put_lines


def sharpness(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    c = g[h // 4:3 * h // 4, w // 4:3 * w // 4]
    # Vùng giữa thu về 320 px ngang: 480p và 720p cùng cảnh so được với nhau
    c = cv2.resize(c, (320, int(320 * c.shape[0] / c.shape[1])), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(c, cv2.CV_64F).var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", default=None)
    ap.add_argument("--source", default=None, help="chỉ số webcam; mặc định lấy từ fusion.cameras hoặc 0")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--fourcc")
    args = ap.parse_args()
    cfg = load_config(args.config)
    src = args.source
    if src is None:
        webcams = [c["source"] for c in cfg.get("fusion", {}).get("cameras", [])
                   if str(c.get("source")).isdigit()]
        src = webcams[0] if webcams else 0
    opts = webcam_options(cfg)
    for k in ("width", "height", "fourcc"):
        if getattr(args, k):
            opts[k] = getattr(args, k)
    if str(src).isdigit() and shutil.which("v4l2-ctl"):
        dev = f"/dev/video{int(src)}"
        print(f"== Định dạng {dev} hỗ trợ (tìm MJPG 1280x720 30 fps):")
        out = subprocess.run(["v4l2-ctl", "-d", dev, "--list-formats-ext"], capture_output=True, text=True).stdout
        print(out.strip() or "(không đọc được)")
        print(f"\n== Control của {dev} (dùng tên ở cột đầu cho camera.v4l2):")
        out = subprocess.run(["v4l2-ctl", "-d", dev, "--list-ctrls-menus"], capture_output=True, text=True).stdout
        print(out.strip() or "(không đọc được)")
    else:
        print("(Cài v4l-utils để xem định dạng/control: sudo apt install v4l-utils)")
    cam = OpenCVSource(src, **opts)
    print(f"\nĐang chạy: {cam.mode}")
    cv2.namedWindow("webcam_check", cv2.WINDOW_NORMAL)
    t_prev, fps, hist = time.monotonic(), 0.0, []
    try:
        while True:
            ok, s = cam.read()
            if not ok:
                raise SystemExit("Mất khung webcam")
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3)
            t_prev = now
            hist = (hist + [sharpness(s.bgr)])[-30:]
            img = s.bgr.copy()
            h, w = img.shape[:2]
            cv2.rectangle(img, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (0, 255, 255), 1)
            put_lines(img, [f"{w}x{h} | {fps:4.1f} fps thuc | {cam.mode}",
                            f"diem net (khung vang): {np.median(hist):6.0f}   do sang TB {img.mean():5.0f}",
                            "s: luu anh | q: thoat"])
            cv2.imshow("webcam_check", img)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("s"):
                f = f"webcam_check_{time.strftime('%H%M%S')}.png"
                cv2.imwrite(f, s.bgr)
                print("Đã lưu", f, f"(điểm nét {np.median(hist):.0f})")
            elif k in (ord("q"), 27):
                break
    finally:
        print(f"Kết thúc: {cam.mode}, {fps:.1f} fps thực, điểm nét {np.median(hist) if hist else 0:.0f}")
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
