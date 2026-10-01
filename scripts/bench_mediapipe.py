#!/usr/bin/env python3
"""So thời gian MediaPipe Pose + Hand trên CPU và GPU của chính máy này (đứng trước camera khi đo).

    python scripts/bench_mediapipe.py                                   # webcam 0, 150 khung mỗi chế độ
    python scripts/bench_mediapipe.py --source "http://127.0.0.1:4747/video" --frames 200
    python scripts/bench_mediapipe.py --source realsense

In ms/khung cho Pose, Hand và cả hai chạy song song, cùng với chế độ thật sự được dùng (GPU lỗi thì quay về CPU).
GPU trên Linux dùng OpenGL ES qua EGL: chạy được cả GPU Intel (Mesa) nhưng chưa chắc nhanh hơn CPU (XNNPACK) với
model nhỏ như MediaPipe; GPU rời NVIDIA thường lợi hơn. Chỉ đổi models.delegate khi số đo cho thấy nhanh hơn.
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from openarm_shadow.config import load_config  # noqa: E402
from openarm_shadow.sources import open_source  # noqa: E402


def bench(cfg, frames, delegate, imgs):
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision
    RM = vision.RunningMode.VIDEO
    dlg = mpt.BaseOptions.Delegate.GPU if delegate == "gpu" else mpt.BaseOptions.Delegate.CPU
    try:
        pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=cfg["models"]["pose"], delegate=dlg), running_mode=RM))
        hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=cfg["models"]["hand"], delegate=dlg), running_mode=RM,
            num_hands=2))
    except Exception as e:
        return f"không khởi tạo được: {type(e).__name__}: {e}"
    tp, th, found = [], [], 0
    for k, bgr in enumerate(imgs[:frames]):
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(bgr[:, :, ::-1]))
        t0 = time.perf_counter()
        r = pose.detect_for_video(img, 33 * k + 1)
        t1 = time.perf_counter()
        h = hand.detect_for_video(img, 33 * k + 1)
        t2 = time.perf_counter()
        tp.append(t1 - t0)
        th.append(t2 - t1)
        found += bool(r.pose_landmarks) and bool(h.hand_landmarks)
    pose.close()
    hand.close()
    skip = min(10, len(tp) // 5)                       # bỏ vài khung đầu (khởi động, biên dịch shader GPU)
    mp_ = 1000 * np.median(tp[skip:])
    mh = 1000 * np.median(th[skip:])
    return (f"Pose {mp_:5.1f} ms | Hand {mh:5.1f} ms | nối tiếp {mp_ + mh:5.1f} ms (~{1000 / (mp_ + mh):4.1f} fps) "
            f"| thấy người + tay {100 * found / max(len(tp), 1):3.0f}% khung")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--config", action="append", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg["camera"]["required_source"] = None
    cap = open_source(args.source, cfg)
    print(f"Thu {args.frames} khung từ {args.source} (đứng trước camera, giơ tay)...")
    imgs = []
    while len(imgs) < args.frames:
        ok, s = cap.read()
        if not ok:
            break
        imgs.append(s.bgr.copy())
    cap.close()
    if not imgs:
        raise SystemExit("Không đọc được khung nào")
    print(f"Ảnh {imgs[0].shape[1]}x{imgs[0].shape[0]}, {len(imgs)} khung (cùng bộ ảnh cho cả hai chế độ)")
    for d in ("cpu", "gpu"):
        print(f"  {d.upper()}: {bench(cfg, args.frames, d, imgs)}")


if __name__ == "__main__":
    main()
