"""SafetyGate bám theo vận tốc + feedforward (safety.velocity_tracking): bớt trễ, không giật, vẫn giữ giới hạn."""
import copy

import numpy as np

from openarm_shadow.config import load_config
from openarm_shadow.core.kinematics import ArmKinematics
from openarm_shadow.safety.gate import SafetyGate


def make(vt):
    cfg = copy.deepcopy(load_config()["safety"])
    cfg["velocity_tracking"] = {"enabled": vt}
    g = SafetyGate({"right": ArmKinematics("right")}, cfg)
    g.reset({"right": np.zeros(8)})
    g.engage(-10.0)
    return g


def motor(q, qd, cmd, dq, kp=10.0, kd=0.7, I=0.004, b=0.02, dt=0.01):
    for _ in range(10):
        qdd = (kp * (cmd - q) + kd * (dq - qd) - b * qd) / I
        qd += qdd * dt / 10
        q += qd * dt / 10
    return q, qd


def run(vt, seconds=6.0, fps=15.0, j=4, amp=np.deg2rad(40), f=0.3):
    g = make(vt)
    q = qd = 0.0
    t, nxt, log = 0.0, 0.0, []
    while t < seconds:
        if t >= nxt - 1e-9:
            tgt = np.full(8, np.nan)
            tgt[j] = amp * np.sin(2 * np.pi * f * t)
            g.set_target({"right": tgt}, t)
            nxt += 1 / fps
        cmd = g.step(0.01, t)["right"]
        dq = g.dq["right"][j] if g.dq is not None else 0.0
        q, qd = motor(q, qd, cmd[j], dq)
        log.append((t, amp * np.sin(2 * np.pi * f * t), cmd[j], q, qd, dq))
        t += 0.01
    return np.array(log), g


def lag(L):
    t, h, _, q = L[200:, 0], L[200:, 1], L[200:, 2], L[200:, 3]
    return min(range(60), key=lambda s: np.mean((q[s:] - h[:len(h) - s]) ** 2)) * 10


def test_tracking_reduces_lag_without_more_jerk():
    old, _ = run(False)
    new, _ = run(True)
    assert lag(new) <= lag(old) - 50, (lag(old), lag(new))
    acc = lambda L: np.sqrt(np.mean(np.diff(L[200:, 4]) ** 2)) / 0.01
    assert acc(new) <= 1.2 * acc(old), (acc(old), acc(new))


def test_tracking_respects_velocity_and_acceleration_limits():
    L, g = run(True, amp=np.deg2rad(80), f=0.8)               # người xoay nhanh hơn giới hạn
    v = np.diff(L[:, 2]) / 0.01
    assert np.max(np.abs(v)) <= g.max_vel[4] + 1e-6
    a = np.diff(v) / 0.01
    assert np.max(np.abs(a)) <= g.max_acc[4] + 1e-3
    assert np.max(np.abs(L[:, 5])) <= g.max_vel[4] + 1e-9    # dq gửi motor cũng trong giới hạn


def test_hold_and_deadman_zero_feedforward():
    g = make(True)
    tgt = np.full(8, np.nan)
    for k in range(5):
        tgt[4] = 0.1 * k
        g.set_target({"right": tgt.copy()}, 0.066 * k)
        g.step(0.01, 0.066 * k + 0.01)
    assert np.any(g.dq["right"] != 0)
    g.step(0.01, 10.0)                                       # mất mục tiêu quá deadman_s
    assert g.status.startswith("hold (dead-man") and np.all(g.dq["right"] == 0)
    g.disengage()
    g.step(0.01, 10.01)
    assert np.all(g.dq["right"] == 0)


def test_disabled_gives_no_feedforward_and_old_behaviour():
    g = make(False)
    tgt = np.full(8, np.nan)
    tgt[0] = 1.0
    g.set_target({"right": tgt}, 0.0)
    cmd = g.step(0.01, 0.0)["right"]
    # kiểu cũ: tiến tối đa max_vel*dt*ramp (mục tiêu đầu tiên sau engage -> ramp bắt đầu lại từ 0,05)
    assert g.dq is None and abs(cmd[0] - g.max_vel[0] * 0.01 * 0.05) < 1e-12


def test_robot_send_passes_dq_with_sign():
    import sys
    import types
    sys.path.insert(0, "tests")
    from test_can_backend import make_fake_openarm_can
    oa = make_fake_openarm_can()

    class MIT:
        def __init__(self, kp, kd, q, dq, tau):
            self.kp, self.kd, self.q, self.dq = kp, kd, q, dq
    oa.MITParam = MIT
    sent = []
    sys.modules["openarm_can"] = oa
    try:
        from openarm_shadow.robot.openarm_can_robot import OpenArmCANRobot
        cfg = load_config()
        cfg["robot"]["feedback_timeout_s"] = 1.0
        cfg["robot"]["urdf_to_motor"]["right"]["sign"] = [1, 1, 1, 1, -1, 1, 1]
        r = OpenArmCANRobot(cfg["robot"], ["right"])
        hw = oa.OpenArm.instances[-1]
        orig = hw.a.mit_control_all
        hw.a.mit_control_all = lambda ps: (sent.append([p for p in ps]), orig(ps))
        q0 = r.connect()
        r.enable()
        dq = np.zeros(7)
        dq[4] = 0.5
        r.send({"right": q0["right"]}, {"right": dq})
        assert abs(sent[-1][4].dq + 0.5) < 1e-12        # dấu motor J5 = -1
        assert all(p.dq == 0.0 for k, p in enumerate(sent[-1]) if k != 4)
        r.send({"right": q0["right"]})                   # không có dq: như cũ
        assert all(p.dq == 0.0 for p in sent[-1])
    finally:
        del sys.modules["openarm_can"]


def test_no_drift_after_tracking_lost():
    """Đang xoay 90°/s thì mất tay (bộ lọc chỉ giữ mục tiêu cũ): lệnh không được vượt mục tiêu cuối."""
    g = make(True)
    t, last = 0.0, 0.0
    for k in range(30):                                       # 2 s bám, J5 90°/s, 15 fps
        tgt = np.full(8, np.nan)
        last = tgt[4] = np.deg2rad(90) * t
        g.set_target({"right": tgt}, t)
        for _ in range(7):
            g.step(0.01, t)
            t += 0.01
    cmd_before = g.cmd["right"][4]
    held = np.full(8, np.nan)
    held[4] = last
    for _ in range(100):                                      # 1 s chỉ có mục tiêu "giữ"
        g.set_target({"right": held}, t, fresh=False)
        g.step(0.01, t)
        t += 0.01
    assert np.degrees(g.cmd["right"][4] - last) <= 0.5, np.degrees(g.cmd["right"][4] - last)
    assert np.all(g.dq["right"] == 0)


def test_standstill_noise_not_amplified():
    """Mục tiêu đứng yên + nhiễu ở 15 fps: nhiễu nhỏ (< min_step) không bị khuếch đại; nhiễu lớn: độ rung (std)
    như kiểu cũ."""
    for amp, metric in ((0.45, np.ptp), (3.0, np.std)):
        res = {}
        for vt in (False, True):
            rng = np.random.default_rng(1)
            g = make(vt)
            t, out = 0.0, []
            for _ in range(300):
                tgt = np.full(8, np.nan)
                tgt[4] = np.deg2rad(amp) * rng.uniform(-1, 1)
                g.set_target({"right": tgt}, t)
                for _ in range(7):
                    out.append(g.step(0.01, t)["right"][4])
                    t += 0.01
            res[vt] = metric(out[300:])
        assert res[True] <= 1.1 * res[False], (amp, res)


def test_velocity_estimate_uses_frame_time_not_arrival_time():
    """Nhận diện ở luồng nền: mục tiêu đến tay luồng chính lệch nhịp, nhưng khung chụp đều 15 fps ->
    vận tốc ước lượng phải theo thời điểm chụp."""
    g = make(True)
    rng = np.random.default_rng(0)
    for k in range(20):
        t_frame = k / 15
        tgt = np.full(8, np.nan)
        tgt[4] = np.deg2rad(60) * t_frame                      # 60°/s đều
        g.set_target({"right": tgt}, t_frame + 0.08 + rng.uniform(0, 0.05), t_frame=t_frame)
    assert abs(np.degrees(g.v_tgt["right"][4]) - 60) < 1.0
