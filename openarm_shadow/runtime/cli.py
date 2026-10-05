"""Tham số dòng lệnh dùng chung cho scripts/shadow.py và node ROS 2 (ros2/openarm_shadow_ros)."""
from __future__ import annotations

import argparse
import os

from ..config import load_config


def parse_args(argv=None, default_robot="sim"):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None,
                    help="realsense (mặc định), multi (nhiều camera, fusion.cameras), replay:DIR (dữ liệu thô đã ghi thay "
                         "camera), chỉ số webcam (vd 0), file video hoặc URL")
    ap.add_argument("--robot", choices=["sim", "openarm", "ros2"], default=default_robot,
                    help="sim: robot mô phỏng; openarm: CAN trực tiếp; ros2: gửi lệnh cho backend OpenArm qua ROS 2")
    ap.add_argument("--config", action="append", default=None,
                    help="file config ghi đè default.yaml; dùng nhiều lần để ghép, file sau thắng")
    ap.add_argument("--mode", choices=["direct", "mirror"], default=None)
    ap.add_argument("--arms", default=None, help="vd: right hoặc right,left")
    ap.add_argument("--record", default=None, help="lưu mục tiêu + lệnh ra file .npz")
    ap.add_argument("--dry-run", action="store_true", help="robot thật: chỉ đọc góc, không bật motor / không gửi lệnh")
    ap.add_argument("--yes", action="store_true", help="không hỏi 'yes' trước khi bắt đầu (chạy từ launch file)")
    ap.add_argument("--headless", action="store_true", help="không mở cửa sổ hiển thị (không có phím điều khiển)")
    ap.add_argument("--replay-intrinsics", default=None,
                    help="--source replay:DIR: file nội tham số RealSense nếu DIR không có intrinsics.yaml")
    ap.add_argument("--replay-start", type=float, default=0.0, help="--source replay:DIR: bỏ qua chừng này giây đầu")
    ap.add_argument("--replay-frames", type=int, default=None, help="--source replay:DIR: số khung tối đa")
    return ap.parse_args(argv)


def main(argv=None, default_robot="sim"):
    os.environ.setdefault("GLOG_minloglevel", "2")   # ẩn log INFO/WARNING của MediaPipe để thấy thông báo thật
    args = parse_args(argv, default_robot)
    cfg = load_config(args.config)
    if args.mode:
        cfg["mapping"]["mode"] = args.mode
    if args.arms:
        cfg["mapping"]["robot_arms"] = args.arms.split(",")
    if args.source is not None:
        # Nguồn chọn tay trên dòng lệnh thắng yêu cầu D455 trong config (vd --source 0 = webcam laptop).
        cfg["camera"]["required_source"] = None
    src = args.source if args.source is not None else cfg["camera"]["index"]
    cfg["replay"] = {"intrinsics": args.replay_intrinsics, "start_s": args.replay_start,
                     "max_frames": args.replay_frames}
    if args.dry_run and args.robot == "sim":
        raise SystemExit("--dry-run chỉ dùng cùng --robot openarm hoặc --robot ros2")
    from .app import run
    run(cfg, src, args.robot, record=args.record, show=not args.headless, dry_run=args.dry_run,
        confirm=not args.yes)
