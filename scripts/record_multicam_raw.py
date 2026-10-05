#!/usr/bin/env python3
"""Ghi RGB/depth thô của tất cả camera để phát triển perception offline.

RGB được lưu MJPG AVI; depth RealSense lưu liên tiếp dạng little-endian uint16 mm; nội tham số RealSense
lưu ở intrinsics.yaml (cần cho scripts/replay_raw.py). Mỗi dòng timestamps.csv ứng với đúng một frame trong AVI
và, nếu có, một ma trận depth trong file .raw.

Ví dụ:
    python scripts/record_multicam_raw.py --config config/local_3cam.yaml --out data/raw_demo --seconds 30
"""
import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openarm_shadow.config import load_config  # noqa: E402
from openarm_shadow.cameras import MultiCameraSource  # noqa: E402


def intrinsics_dict(intr):
    """Nội tham số RealSense (rs.intrinsics hoặc dict) -> dict ghi được ra YAML."""
    if isinstance(intr, dict):
        return {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in intr.items()}
    return {"width": int(intr.width), "height": int(intr.height), "fx": float(intr.fx), "fy": float(intr.fy),
            "ppx": float(intr.ppx), "ppy": float(intr.ppy), "model": str(intr.model),
            "coeffs": [float(c) for c in intr.coeffs]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--countdown", type=int, default=5)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Thư mục đích đã có dữ liệu: {out}")
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    cams_cfg = cfg["fusion"]["cameras"]
    names = [c["name"] for c in cams_cfg]
    src = MultiCameraSource(cfg)
    writers, depth_files, csv_files, csv_writers = {}, {}, {}, {}
    frame_counts = {n: 0 for n in names}
    last_t = [-np.inf] * len(names)
    shapes = {}
    intrinsics = {}
    t_wall = time.time()
    try:
        ok, _ = src.read(timeout=10.0)
        if not ok:
            raise SystemExit("Không đọc được đủ camera:\n" + (src.error or ""))
        for k in range(args.countdown, 0, -1):
            print(f"Bắt đầu ghi sau {k} giây...", flush=True)
            time.sleep(1.0)
        with src.cond:
            for i, b in enumerate(src.buf):
                if b:
                    last_t[i] = b[-1][0]
        t0 = time.monotonic()
        print(f"ĐANG GHI {args.seconds:g} giây vào {out}", flush=True)
        while time.monotonic() - t0 < args.seconds:
            with src.cond:
                bufs = [list(b) for b in src.buf]
            for i, buf in enumerate(bufs):
                name = names[i]
                for t_corrected, sample in buf:
                    if t_corrected <= last_t[i]:
                        continue
                    last_t[i] = t_corrected
                    h, w = sample.bgr.shape[:2]
                    if name not in writers:
                        cam_dir = out / name
                        cam_dir.mkdir(parents=True, exist_ok=True)
                        writer = cv2.VideoWriter(str(cam_dir / "rgb.avi"), cv2.VideoWriter_fourcc(*"MJPG"),
                                                 float(cfg["camera"].get("fps", 30)), (w, h))
                        if not writer.isOpened():
                            raise RuntimeError(f"Không mở được VideoWriter cho {name}")
                        writers[name] = writer
                        f = (cam_dir / "timestamps.csv").open("w", newline="")
                        csv_files[name] = f
                        csv_writers[name] = csv.writer(f)
                        csv_writers[name].writerow(["frame", "host_arrival_s", "device_timestamp_s"])
                        shapes[name] = {"rgb_width": w, "rgb_height": h, "depth_width": None,
                                        "depth_height": None, "depth_dtype": None, "depth_unit": None}
                        if sample.depth_m is not None:
                            depth_files[name] = (cam_dir / "depth_u16_mm.raw").open("wb")
                            shapes[name].update(depth_width=sample.depth_m.shape[1], depth_height=sample.depth_m.shape[0],
                                                depth_dtype="<u2", depth_unit="mm")
                        if sample.intrinsics is not None:
                            intrinsics[name] = intrinsics_dict(sample.intrinsics)
                    writers[name].write(sample.bgr)
                    if sample.depth_m is not None:
                        depth_mm = np.clip(np.rint(sample.depth_m * 1000.0), 0, 65535).astype("<u2")
                        depth_files[name].write(depth_mm.tobytes(order="C"))
                    host_arrival = t_corrected + src.latency[i]
                    csv_writers[name].writerow([frame_counts[name], f"{host_arrival:.9f}",
                                                "" if sample.timestamp_s is None else f"{sample.timestamp_s:.6f}"])
                    frame_counts[name] += 1
            time.sleep(0.003)
        elapsed = time.monotonic() - t0
    finally:
        src.close()
        for w in writers.values():
            w.release()
        for f in depth_files.values():
            f.close()
        for f in csv_files.values():
            f.close()

    for p in args.config:
        shutil.copy2(p, out / Path(p).name)
    calib = Path(cfg["fusion"]["calib_file"])
    if calib.exists():
        shutil.copy2(calib, out / calib.name)
    if intrinsics:
        # Nội tham số RGB (depth đã align vào RGB): cần để chạy lại bằng scripts/replay_raw.py
        (out / "intrinsics.yaml").write_text(yaml.safe_dump({"cameras": intrinsics}, sort_keys=False))
    (out / "effective_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    meta = {"created_unix_s": t_wall, "duration_s": elapsed, "requested_seconds": args.seconds,
            "cameras": {n: {**shapes.get(n, {}), "frames": frame_counts[n],
                              "measured_fps": frame_counts[n] / elapsed} for n in names},
            "note": "RGB frame k and depth matrix k share row k in timestamps.csv. "
                    "Read depth with np.fromfile(..., dtype='<u2').reshape(-1,H,W)."}
    (out / "metadata.yaml").write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True))
    print("HOÀN TẤT:", out)
    for n in names:
        print(f"  {n}: {frame_counts[n]} frame, {frame_counts[n] / elapsed:.1f} fps")


if __name__ == "__main__":
    main()
