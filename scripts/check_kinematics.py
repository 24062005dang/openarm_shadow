#!/usr/bin/env python3
"""Kiểm tra model động học OpenArm v1.0 và bộ retarget, không cần camera hay robot.

In ra: trục các khớp ở q = 0, tích vô hướng giữa hai trục liên tiếp (phải = 0),
và thử ngược: lấy tư thế robot ngẫu nhiên -> hướng các đoạn tay -> retarget -> so sánh.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openarm_shadow.core.kinematics import ArmKinematics
from openarm_shadow.mapping.retarget import ArmRetargeter

rng = np.random.default_rng(1)
for side in ("right", "left"):
    kin = ArmKinematics(side)
    rt = ArmRetargeter(kin)
    q0 = np.zeros(7)
    print(f"== tay {side}")
    for i in range(1, 8):
        a = kin.axis_world(q0, i)
        j = kin.joints[i - 1]
        print(f"  J{i}: trục world ở q=0 = {np.round(a, 3)}   giới hạn URDF "
              f"[{np.rad2deg(j.lower):7.1f}, {np.rad2deg(j.upper):7.1f}] độ")
    dots = [kin.axis_world(q0, i) @ kin.axis_world(q0, i + 1) for i in range(1, 7)]
    print("  tích vô hướng trục liên tiếp:", np.round(dots, 6), "(phải toàn 0)")
    k = kin.keypoints(q0)
    print("  vai/khuỷu/cổ tay ở q=0:", {n: np.round(v, 3).tolist() for n, v in k.items()})
    worst = 0.0
    for _ in range(500):
        q = rng.uniform(kin.lower, kin.upper)
        q[3] = rng.uniform(0.4, 2.3)
        u = kin.axis_world(q, 3) * kin.limb_sign[3]
        l = kin.axis_world(q, 5) * kin.limb_sign[5]
        H = kin.R0(q, 7) @ rt.R_offset.T
        qs, _ = rt.solve(u, l, H, q + rng.normal(0, 0.03, 7))
        worst = max(worst, np.abs(qs - q).max())
    print(f"  thử ngược 500 tư thế: sai lệch góc lớn nhất = {np.rad2deg(worst):.2e} độ")
