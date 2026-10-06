#!/usr/bin/env python3
"""Đo độ trễ thật của từng camera so với camera tham chiếu (camera đầu trong fusion.cameras) -> fusion.cameras[i].latency_s.

    python scripts/measure_camera_latency.py --config config/fusion_3cam.yaml

Trong lúc đo (mặc định 12 s): đứng sao cho MỌI camera thấy, vẫy / giơ-hạ tay NHANH rồi dừng hẳn ~1 s, lặp lại (chuyển
động ngắt quãng cho đỉnh tương quan rõ). Mỗi camera ghi "mức chuyển động" (hiệu 2 khung liên tiếp, ảnh thu nhỏ) theo
thời điểm khung ĐẾN máy; tương quan chéo với camera tham chiếu ra độ trễ tương đối. Không cần MediaPipe.
latency_s đề xuất = latency_s camera tham chiếu + độ trễ đo được. Hệ số tương quan < 0,5: đo lại, cử động mạnh hơn.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openarm_shadow.camera.calibration import estimate_time_offset, motion_energy
from openarm_shadow.config import load_config  # noqa: E402
from openarm_shadow.camera.multi_source import MultiCameraSource


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", default=None)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--max-lag", type=float, default=0.3, help="độ trễ lớn nhất tìm kiếm (s)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cams = cfg["fusion"]["cameras"]
    names = [c["name"] for c in cams]
    old = [float(c.get("latency_s", 0.0)) for c in cams]
    src = MultiCameraSource(cfg)
    try:
        ok, _ = src.read(timeout=10.0)
        if not ok:
            raise SystemExit("Không đọc được đủ camera:\n" + (src.error or ""))
        src.latency = [0.0] * len(names)           # thời điểm khung = lúc đến máy (không bù latency_s cũ)
        sig = [([], []) for _ in names]
        prev = [None] * len(names)
        last_t = [-np.inf] * len(names)             # khung mới = thời điểm lớn hơn khung đã xử lý
        for k in (3, 2, 1):
            print(f"Bắt đầu sau {k}... (vẫy tay nhanh rồi dừng, lặp lại)")
            time.sleep(1.0)
        print(f"ĐO {args.seconds:.0f} s")
        t_end = time.monotonic() + args.seconds
        while time.monotonic() < t_end:
            with src.cond:
                bufs = [list(b) for b in src.buf]
            for i, buf in enumerate(bufs):
                for t, s in buf:
                    if t <= last_t[i]:
                        continue
                    last_t[i] = t
                    small = cv2.resize(cv2.cvtColor(s.bgr, cv2.COLOR_BGR2GRAY), (160, 90),
                                       interpolation=cv2.INTER_AREA).astype(np.float32)
                    if prev[i] is not None:
                        sig[i][0].append(t)
                        sig[i][1].append(motion_energy(prev[i], small))
                    prev[i] = small
            time.sleep(0.004)
    finally:
        src.close()
    t_ref, x_ref = sig[0]
    print(f"\n{names[0]}: tham chiếu, {len(t_ref) / args.seconds:.1f} fps, latency_s {old[0]:.3f}")
    for i in range(1, len(names)):
        t, x = sig[i]
        order = np.argsort(t)
        lag, corr = estimate_time_offset(t_ref, x_ref, np.asarray(t)[order], np.asarray(x)[order], args.max_lag)
        fps = len(t) / args.seconds
        if not np.isfinite(lag):
            print(f"{names[i]}: không đủ dữ liệu ({fps:.1f} fps)")
            continue
        note = "" if corr >= 0.5 else "  <- tương quan thấp: đo lại, cử động mạnh và ngắt quãng hơn"
        print(f"{names[i]}: {fps:.1f} fps, trễ hơn {names[0]} {1000 * lag:+.0f} ms (tương quan {corr:.2f}), "
              f"latency_s đang đặt {old[i]:.3f} -> đề xuất {old[0] + lag:.3f}{note}")


if __name__ == "__main__":
    main()
