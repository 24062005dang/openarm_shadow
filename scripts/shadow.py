#!/usr/bin/env python3
"""Chạy teleop bắt chước tay.

    python scripts/shadow.py                      # D455 RGB-D, robot mô phỏng
    python scripts/shadow.py --source 0           # webcam laptop (không có depth), robot mô phỏng
    python scripts/shadow.py --source multi --config config/fusion_2cam.yaml   # 2 camera, fusion
    python scripts/shadow.py --robot openarm --dry-run   # đọc robot thật, motor TẮT (kiểm tra chiều khớp)
    python scripts/shadow.py --robot openarm --config config/first_real.yaml --arms right   # lần chạy thật đầu
    python scripts/shadow.py --robot ros2 --dry-run      # backend OpenArm qua ROS 2: chỉ đọc joint_states
    python scripts/shadow.py --config my.yaml --record run1.npz
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openarm_shadow.runtime.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
