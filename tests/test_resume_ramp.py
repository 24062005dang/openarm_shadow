"""Khớp đứng yên (mất tay / hướng tay chưa chắc) rồi chạy lại: SafetyGate tăng tốc mềm riêng khớp đó, không lao vụt
tới mục tiêu mới (lỗi cổ tay tự xoay nhanh khi tay để ngang rồi thấy lại ở hướng khác)."""
import copy

import numpy as np
import pytest

from openarm_shadow.config import load_config
from openarm_shadow.core.kinematics import ArmKinematics
from openarm_shadow.safety import SafetyGate


def make(vt=False):
    cfg = copy.deepcopy(load_config()["safety"])
    cfg["velocity_tracking"] = {"enabled": vt}
    g = SafetyGate({"right": ArmKinematics("right")}, cfg)
    g.reset({"right": np.zeros(8)})
    g.engage(-10.0)
    return g


def held_flags(wrist_held):
    h = np.zeros(8, bool)
    h[4:7] = wrist_held
    return {"right": h}


def drive(g, t0, seconds, tgt, wrist_held, fps=15.0, dt=0.01):
    """Chạy gate từ t0, trả (t, góc J5 mỗi nhịp)."""
    t, nxt, ts, qs = t0, t0, [], []
    while t < t0 + seconds - 1e-9:
        if t >= nxt - 1e-9:
            g.set_target({"right": tgt.copy()}, t, fresh=True, held=held_flags(wrist_held))
            nxt += 1 / fps
        q = g.step(dt, t)["right"]
        ts.append(t)
        qs.append(q[4])
        t += dt
    return np.array(ts), np.array(qs)


@pytest.mark.parametrize("vt", [False, True])
def test_wrist_resumes_slowly_after_hold(vt):
    g = make(vt)
    tgt = np.zeros(8)
    drive(g, 0.0, 2.0, tgt, False)                       # qua hết tăng tốc mềm lúc engage
    drive(g, 2.0, 0.6, tgt, True)                        # mất hướng tay 0,6 s: cổ tay giữ
    far = tgt.copy()
    far[4] = np.deg2rad(80)                              # thấy lại ở hướng xa 80°
    ts, qs = drive(g, 2.6, 0.4, far, False)
    vmax = g.max_vel[4]
    v = np.abs(np.diff(qs)) / 0.01
    assert v.max() < 0.5 * vmax                          # 0,4 s đầu: còn chậm hơn hẳn tốc độ tối đa
    assert qs[-1] > 0                                    # nhưng vẫn đi đúng chiều, không đứng hẳn
    _, qs2 = drive(g, 3.0, 3.0, far, False)
    assert abs(qs2[-1] - far[4]) < np.deg2rad(2)         # cuối cùng vẫn tới


def test_short_hold_does_not_slow_down():
    g = make()
    tgt = np.zeros(8)
    drive(g, 0.0, 2.0, tgt, False)                       # qua hết tăng tốc mềm lúc engage
    drive(g, 2.0, 0.1, tgt, True)                        # giữ 0,1 s < resume_after_s: không coi là chạy lại
    far = tgt.copy()
    far[4] = np.deg2rad(60)
    _, qs = drive(g, 2.1, 0.3, far, False)
    v = np.abs(np.diff(qs)) / 0.01
    assert v.max() > 0.9 * g.max_vel[4]


def test_ramp_is_per_joint():
    g = make()
    tgt = np.zeros(8)
    drive(g, 0.0, 2.0, tgt, False)                       # qua hết tăng tốc mềm lúc engage
    drive(g, 2.0, 0.6, tgt, True)                        # chỉ cổ tay giữ
    g.set_target({"right": tgt}, 2.7, held=held_flags(False))
    r = g.joint_ramp("right", 2.71)
    assert np.all(r[:4] == 1.0) and np.all(r[4:7] < 0.1)


def test_not_fresh_counts_as_held():
    g = make()
    tgt = np.zeros(8)
    drive(g, 0.0, 2.0, tgt, False)                       # qua hết tăng tốc mềm lúc engage
    for k in range(10):                                  # mất người 0,5 s (pipeline báo không có giá trị mới)
        g.set_target({"right": tgt}, 2.0 + 0.05 * k, fresh=False)
    g.set_target({"right": tgt}, 2.5, fresh=True, held={"right": np.zeros(8, bool)})
    assert np.all(g.joint_ramp("right", 2.51)[:7] < 0.1)
