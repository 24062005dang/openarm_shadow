#!/usr/bin/env python3
"""Chạy teleop bắt chước tay.

    python scripts/shadow.py                      # webcam 0, robot mô phỏng (an toàn, nên chạy trước)
    python scripts/shadow.py --source video.mp4   # chạy trên video quay sẵn
    python scripts/shadow.py --robot openarm      # OpenArm thật (đọc docs/SAFETY.md trước)
    python scripts/shadow.py --config my.yaml --record run1.npz
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openarm_shadow.app import run
from openarm_shadow.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None, help="chỉ số webcam, đường dẫn video hoặc URL luồng")
    ap.add_argument("--robot", choices=["sim", "openarm"], default="sim")
    ap.add_argument("--config", default=None)
    ap.add_argument("--mode", choices=["direct", "mirror"], default=None)
    ap.add_argument("--arms", default=None, help="vd: right hoặc right,left")
    ap.add_argument("--record", default=None, help="lưu mục tiêu + lệnh ra file .npz")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.mode:
        cfg["mapping"]["mode"] = args.mode
    if args.arms:
        cfg["mapping"]["robot_arms"] = args.arms.split(",")
    src = args.source if args.source is not None else cfg["camera"]["index"]
    run(cfg, src, args.robot, record=args.record)


if __name__ == "__main__":
    main()
