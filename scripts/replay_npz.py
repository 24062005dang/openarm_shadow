#!/usr/bin/env python3
"""Phát lại quỹ đạo .npz (từ offline_retarget.py hoặc --record) qua SafetyGate.

    python scripts/replay_npz.py taichi_openarm.npz                   # mô phỏng, có hình
    python scripts/replay_npz.py taichi_openarm.npz --robot openarm --speed 0.5
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openarm_shadow.app import park
from openarm_shadow.config import load_config
from openarm_shadow.kinematics import ArmKinematics
from openarm_shadow.robot import make_robot
from openarm_shadow.safety import SafetyGate
from openarm_shadow.viz import draw_robot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--robot", choices=["sim", "openarm"], default="sim")
    ap.add_argument("--config", default=None)
    ap.add_argument("--speed", type=float, default=1.0, help="hệ số tốc độ phát (0.5 = chậm một nửa)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    d = np.load(args.npz)
    sides = [s for s in ("right", "left") if f"q_{s}" in d or f"target_{s}" in d]
    traj = {s: d[f"q_{s}"] if f"q_{s}" in d else d[f"target_{s}"] for s in sides}
    t = d["t"] - d["t"][0]
    kins = {s: ArmKinematics(s) for s in sides}
    robot = make_robot(args.robot, cfg, sides)
    gate = SafetyGate(kins, cfg["safety"])
    gate.reset(robot.connect())
    if args.robot == "openarm":
        if input("ROBOT THẬT. Gõ 'yes' để bật motor: ").strip().lower() != "yes":
            raise SystemExit("Huỷ.")
        robot.enable()
    dt = 1.0 / cfg["robot"]["control_hz"]
    t0 = time.monotonic()
    gate.engage(t0)
    try:
        while True:
            now = time.monotonic()
            tr = (now - t0) * args.speed
            if tr > t[-1]:
                break
            k = int(np.searchsorted(t, tr))
            gate.set_target({s: traj[s][min(k, len(t) - 1)] for s in sides}, now)
            cmd = gate.step(dt, now)
            robot.send(cmd)
            if args.robot == "sim":
                cv2.imshow("replay", draw_robot(kins, cmd, title=f"t = {tr:5.1f} s | {gate.status}"))
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            time.sleep(max(0.0, dt - (time.monotonic() - now)))
    finally:
        if args.robot == "openarm":
            park(robot, gate, np.deg2rad(cfg["robot"]["rest_pose_deg"]), cfg["robot"]["park_vel_deg_s"])
        robot.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
