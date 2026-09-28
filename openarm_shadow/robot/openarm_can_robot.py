"""Backend OpenArm v1.0 thật qua openarm_can (CAN-FD, SavvyCAN/PEAK -> can0/can1).

CHƯA CHẠY TRÊN ROBOT THẬT. Trước khi dùng phải làm các bước trong docs/SAFETY.md, nhất là
kiểm tra quy ước góc URDF ↔ góc motor (robot.urdf_to_motor trong config).

API openarm_can dùng ở đây đã đối chiếu với enactic/openarm_can commit f340d4b (16/09/2026).
"""
from __future__ import annotations

import time

import numpy as np

MOTOR_TYPES = ["DM8009", "DM8009", "DM4340", "DM4340", "DM4310", "DM4310", "DM4310"]
SEND_IDS = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07]
RECV_IDS = [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17]


class RobotFault(RuntimeError):
    pass


class _Arm:
    def __init__(self, side, rcfg):
        import openarm_can as oa
        self.oa = oa
        self.side = side
        self.iface = rcfg["interfaces"][side]
        m = rcfg["urdf_to_motor"][side]
        self.sign = np.asarray(m["sign"], float)
        self.offset = np.deg2rad(np.asarray(m["offset_deg"], float))
        lim = np.deg2rad(np.asarray(rcfg["motor_limits_deg"][side], float))
        self.mlo, self.mhi = lim[:, 0], lim[:, 1]
        self.kp = np.asarray(rcfg["kp"], float)
        self.kd = np.asarray(rcfg["kd"], float)
        g = rcfg["gripper"]
        self.grip_on = bool(g["enabled"])
        self.g_open, self.g_closed = np.deg2rad(g["open_deg"]), np.deg2rad(g["closed_deg"])
        self.g_kp, self.g_kd = float(g["kp"]), float(g["kd"])
        self.stale_s = float(rcfg.get("feedback_timeout_s", 0.1))

        self.arm = oa.OpenArm(self.iface, True)
        self.arm.init_arm_motors([getattr(oa.MotorType, t) for t in MOTOR_TYPES], SEND_IDS, RECV_IDS)
        self.arm.init_gripper_motor(oa.MotorType.DM4310, 0x08, 0x18)
        self.arm.set_callback_mode_all(oa.CallbackMode.STATE)
        self.q_motor = np.full(7, np.nan)
        self.g_motor = np.nan

    # ---- đổi đơn vị ----
    def to_motor(self, q_urdf):
        return self.sign * np.asarray(q_urdf) + self.offset

    def to_urdf(self, q_motor):
        return (np.asarray(q_motor) - self.offset) * self.sign

    def grip_to_motor(self, f):
        return self.g_closed + f * (self.g_open - self.g_closed)

    def grip_from_motor(self, g):
        span = self.g_open - self.g_closed
        return float(np.clip((g - self.g_closed) / span, 0, 1)) if abs(span) > 1e-6 else 0.5

    # ---- đọc ----
    def _update(self):
        self.q_motor = np.array([m.get_position() for m in self.arm.get_arm().get_motors()])
        gm = self.arm.get_gripper().get_motors()
        self.g_motor = gm[0].get_position() if gm else np.nan

    def check_fresh(self):
        comp = self.arm.get_arm()
        if hasattr(comp, "get_link_stats"):
            stale = [i + 1 for i in range(7) if comp.get_link_stats(i).seconds_since_response() > self.stale_s]
            if stale:
                raise RobotFault(f"{self.side}: mất phản hồi khớp {stale}")
        if not np.all(np.isfinite(self.q_motor)):
            raise RobotFault(f"{self.side}: đọc góc không hợp lệ {self.q_motor}")

    def refresh(self):
        self.arm.refresh_all()
        self.arm.recv_all(2000)
        self._update()

    def state(self):
        return np.append(self.to_urdf(self.q_motor), self.grip_from_motor(self.g_motor))


class OpenArmCANRobot:
    def __init__(self, rcfg: dict, sides):
        self.rcfg = rcfg
        self.sides = list(sides)
        self.arms = {s: _Arm(s, rcfg) for s in self.sides}
        self.enabled = False
        self.t_enable = 0.0
        self.gain_ramp_s = float(rcfg.get("gain_ramp_s", 1.0))
        self.gravity = None
        gc = rcfg.get("gravity_comp", {})
        if gc.get("enabled"):
            from .gravity import GravityModel
            self.gravity = GravityModel(gc["urdf"], self.sides)

    def connect(self):
        """Đọc tư thế khi motor còn tắt. Trả về trạng thái theo góc URDF."""
        for _ in range(3):
            for a in self.arms.values():
                a.refresh()
            time.sleep(0.02)
        for a in self.arms.values():
            a.check_fresh()
        return self.read()

    def read(self):
        return {s: a.state() for s, a in self.arms.items()}

    def enable(self):
        for a in self.arms.values():
            a.arm.enable_all()
            time.sleep(0.05)
            a.arm.recv_all(2000)
        self.enabled, self.t_enable = True, time.monotonic()

    def send(self, cmd):
        if not self.enabled:
            raise RobotFault("send() khi chưa enable")
        ramp = min(1.0, (time.monotonic() - self.t_enable) / self.gain_ramp_s)
        tau = self.gravity.torques({s: np.asarray(cmd[s][:7]) for s in self.sides}) if self.gravity else None
        oa = next(iter(self.arms.values())).oa
        for s, a in self.arms.items():
            q_m = np.clip(a.to_motor(cmd[s][:7]), a.mlo, a.mhi)
            ff = np.zeros(7) if tau is None else a.sign * tau[s]
            a.arm.get_arm().mit_control_all(
                [oa.MITParam(float(a.kp[i] * ramp), float(a.kd[i]), float(q_m[i]), 0.0, float(ff[i]))
                 for i in range(7)])
            if a.grip_on and np.isfinite(cmd[s][7]):
                a.arm.get_gripper().mit_control_all(
                    [oa.MITParam(a.g_kp * ramp, a.g_kd, float(a.grip_to_motor(cmd[s][7])), 0.0, 0.0)])
            a.arm.recv_all(1000)
            a._update()
            a.check_fresh()

    def relax(self, seconds=1.0):
        """Chuyển sang giảm chấn (kp = 0) rồi tắt motor. Gọi khi tay đã về tư thế nghỉ."""
        if not self.enabled:
            return
        oa = next(iter(self.arms.values())).oa
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end:
            for a in self.arms.values():
                a.arm.get_arm().mit_control_all(
                    [oa.MITParam(0.0, float(a.kd[i]), float(a.q_motor[i]), 0.0, 0.0) for i in range(7)])
                a.arm.recv_all(1000)
                a._update()
            time.sleep(0.01)
        for a in self.arms.values():
            a.arm.disable_all()
            a.arm.recv_all(2000)
        self.enabled = False

    def close(self):
        try:
            self.relax()
        finally:
            self.enabled = False
