#!/usr/bin/env python3
"""Chạy thử mô phỏng KHÔNG cần camera, model MediaPipe hay robot.

Một "người giả lập" làm lần lượt vài động tác (nâng tay trước, dang tay, gập khuỷu, vẫy, bắt chéo hai tay (SafetyGate chặn),
che tay một lúc). Điểm mốc có nhiễu như MediaPipe. Luồng xử lý giống hệt khi chạy thật:
ShadowPipeline (retarget + lọc) -> SafetyGate -> SimRobot, rồi vẽ hình que robot.

    python scripts/demo_sim.py                    # mở cửa sổ, q/Esc để thoát
    python scripts/demo_sim.py --out demo.mp4     # không mở cửa sổ, ghi ra video
    python scripts/demo_sim.py --mode mirror

Hình: nét mảnh xám = mục tiêu sau retarget + lọc; nét đậm = lệnh sau SafetyGate (thứ gửi xuống robot).
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openarm_shadow.config import load_config
from openarm_shadow.perception import ArmObs, Frame
from openarm_shadow.retarget.pipeline import ShadowPipeline
from openarm_shadow.robot import make_robot
from openarm_shadow.safety import SafetyGate
from openarm_shadow.viz import draw_robot, put_lines

D = np.deg2rad
# (thời điểm s, động tác đang làm để tới tư thế này, tư thế tay phải J1..J7 độ; None = tay bị che).
# Tay trái đối xứng (nhân dấu mirror).
KEYS = [
    (0.0, "tay tha xuoi", [0, 0, 0, 0, 0, 0, 0]),
    (1.0, "tay tha xuoi", [0, 0, 0, 0, 0, 0, 0]),
    (3.0, "nang tay ra truoc", [60, 0, 0, 10, 0, 0, 0]),
    (5.5, "dang tay ngang", [0, 75, 0, 10, 0, 0, 0]),
    (8.0, "gap khuyu truoc nguc", [40, 20, 0, 100, 0, 0, 0]),
    (10.5, "xoay canh tay, cang tay, co tay", [40, 20, 30, 90, 40, 25, 30]),
    (13.0, "bat cheo hai tay truoc nguc: SafetyGate chan", [70, -9, -40, 110, 0, 0, 0]),
    (14.0, "giu", [70, -9, -40, 110, 0, 0, 0]),
    (15.5, "che tay 1.5 s: robot giu nguyen", None),
    (16.0, "thay lai tay", [70, -9, -40, 110, 0, 0, 0]),
    (18.5, "tay tha xuoi", [0, 0, 0, 0, 0, 0, 0]),
    (19.5, "tay tha xuoi", [0, 0, 0, 0, 0, 0, 0]),
]
MIRROR = np.array([-1, -1, -1, 1, -1, -1, -1])


def human_pose(t):
    """Tư thế 'người' (theo góc khớp robot) ở thời điểm t, nội suy mượt giữa các mốc; None = che tay."""
    for (t0, _, a), (t1, name, b) in zip(KEYS[:-1], KEYS[1:]):
        if t0 <= t < t1:
            if b is None:
                return name, None
            if a is None:
                a = b
            x = (t - t0) / (t1 - t0)
            x = x * x * (3 - 2 * x)
            qr = D(np.asarray(a) + x * (np.asarray(b) - np.asarray(a)))
            return name, {"right": qr, "left": MIRROR * qr}
    qr = D(np.asarray(KEYS[-1][2]))
    return KEYS[-1][1], {"right": qr, "left": MIRROR * qr}


def fake_frame(pipe, q_true, t, rng, noise=0.004):
    arms = {}
    for s, kin in pipe.kins.items():
        if q_true is None:
            arms[s] = ArmObs(None, None, None, None, None, {"upper": 0.0, "fore": 0.0, "hand": 0.0})
            continue
        q = q_true[s]
        s_ = 1.3 * kin.keypoints(q)["shoulder"]                 # người to hơn robot: chỉ hướng là quan trọng
        e_ = s_ + 0.30 * kin.axis_world(q, 3) * kin.limb_sign[3] + rng.normal(0, noise, 3)
        w_ = e_ + 0.27 * kin.axis_world(q, 5) * kin.limb_sign[5] + rng.normal(0, noise, 3)
        H = kin.R0(q, 7) @ pipe.rt[s].R_offset.T
        arms[s] = ArmObs(s_, e_, w_, H, 0.5 + 0.4 * np.sin(t), {"upper": 0.95, "fore": 0.95, "hand": 0.9})
    return Frame(arms, None, [], np.eye(3), t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--mode", choices=["direct", "mirror"], default=None)
    ap.add_argument("--fps", type=float, default=30.0, help="tốc độ khung 'camera' giả lập")
    ap.add_argument("--out", default=None, help="ghi video .mp4 thay vì mở cửa sổ")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.mode:
        cfg["mapping"]["mode"] = args.mode
    pipe = ShadowPipeline(cfg)
    robot = make_robot("sim", cfg, pipe.robot_sides)
    q_meas = robot.connect()
    gate = SafetyGate(pipe.kins, cfg["safety"])
    gate.reset(q_meas)
    pipe.seed(q_meas)
    gate.engage(0.0)

    rng = np.random.default_rng(0)
    dt_cam, n_ctl = 1.0 / args.fps, max(1, int(round(cfg["robot"]["control_hz"] / args.fps)))
    writer = None
    if args.out:
        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (640, 640))
    t, t_end = 0.0, KEYS[-1][0]
    worst = {s: 0.0 for s in pipe.robot_sides}
    statuses, blocked = {}, 0
    wall0 = time.monotonic()
    while t < t_end:
        name, q_true = human_pose(t)
        targets = pipe.step(fake_frame(pipe, q_true, t, rng))
        gate.set_target(targets, t)
        for k in range(n_ctl):                               # vòng điều khiển chạy nhanh hơn camera
            cmd = gate.step(dt_cam / n_ctl, t + (k + 1) * dt_cam / n_ctl)
            robot.send(cmd)
        if q_true is not None and gate.status == "follow":
            for s in pipe.robot_sides:
                src = pipe.human_side_for(s)
                qt = q_true[src] if src == s else MIRROR * q_true[src]
                worst[s] = max(worst[s], float(np.nanmax(np.rad2deg(np.abs(targets[s][:4] - qt[:4])))))
        statuses[gate.status.split(" (")[0].split(" ")[0]] = statuses.get(gate.status.split(" (")[0].split(" ")[0], 0) + 1
        if "va chạm" in gate.status:
            blocked += 1
        img = draw_robot(pipe.kins, robot.read(), size=(640, 640), q_target=targets,
                         title=f"t = {t:4.1f} s | {name}")
        put_lines(img, [f"SafetyGate: {gate.status}"], org=(10, 46), color=(30, 30, 200))
        if writer is not None:
            writer.write(img)
        else:
            cv2.imshow("openarm_shadow demo (khong camera)", img)
            if cv2.waitKey(max(1, int(1000 * (dt_cam - (time.monotonic() - wall0) % dt_cam)))) & 0xFF in (ord("q"), 27):
                break
        t += dt_cam
    if writer is not None:
        writer.release()
        print("Đã ghi", args.out)
    cv2.destroyAllWindows()
    print("Sai lệch lớn nhất J1–J4 giữa mục tiêu và tư thế người (lúc follow, độ):",
          {s: round(v, 1) for s, v in worst.items()})
    print(f"Số khung SafetyGate chặn vì hai tay sắp va nhau: {blocked}")


if __name__ == "__main__":
    main()
