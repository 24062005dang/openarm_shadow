#!/usr/bin/env python3
"""Video quay sẵn -> quỹ đạo góc khớp OpenArm (.npz). Dùng để thử pipeline không cần robot và thu demo cho IL.

    python scripts/offline_retarget.py demo.mp4 -o demo.npz --show

File .npz: fps, t (s), q_right/q_left (N, 8: 7 góc URDF rad + độ mở kẹp 0..1, NaN = mất tracking).
Phát lại lên robot bằng scripts/replay_npz.py (luôn đi qua SafetyGate).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openarm_shadow.config import load_config
from openarm_shadow.vision.perception import Perception
from openarm_shadow.mapping.pipeline import ShadowPipeline
from openarm_shadow.display.viz import draw_human, draw_robot, side_by_side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--mode", choices=["direct", "mirror"], default=None)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.mode:
        cfg["mapping"]["mode"] = args.mode
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    perc = Perception(cfg["models"]["pose"], cfg["models"]["hand"], min_conf=cfg["models"]["min_conf"])
    pipe = ShadowPipeline(cfg)
    ts, qs = [], {s: [] for s in pipe.robot_sides}
    i = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        t = i / fps
        fr = perc.process(img, t=t)
        tg = pipe.step(fr)
        ts.append(t)
        for s in pipe.robot_sides:
            qs[s].append(tg[s])
        if args.show:
            view = side_by_side(draw_human(img.copy(), fr), draw_robot(pipe.kins, tg))
            cv2.imshow("offline_retarget", view)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
        i += 1
    perc.close()
    np.savez(args.out, fps=fps, t=np.array(ts), **{f"q_{s}": np.array(v) for s, v in qs.items()})
    print(f"Đã lưu {args.out}: {len(ts)} khung, {fps:.1f} fps")


if __name__ == "__main__":
    main()
