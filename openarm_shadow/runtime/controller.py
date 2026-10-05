"""Luồng điều khiển chạy đều control_hz (SafetyGate -> robot) và đưa tay về tư thế nghỉ."""
from __future__ import annotations

import threading
import time

import numpy as np


class Controller(threading.Thread):
    def __init__(self, robot, gate, hz):
        super().__init__(daemon=True)
        self.robot, self.gate, self.dt = robot, gate, 1.0 / hz
        self.lock = threading.Lock()
        self.running = True
        self.error = None
        self.cmd = None
        self.trace = None             # list -> ghi (t, lệnh, đo) mỗi nhịp điều khiển

    def run(self):
        t_prev = time.monotonic()
        follows = bool(getattr(self.robot, "follows_measured", False))
        try:
            while self.running:
                now = time.monotonic()
                with self.lock:
                    if follows and not self.gate.engaged:
                        self.gate.sync(self.robot.read())     # robot do backend giữ: lệnh bám theo góc đo
                    cmd = self.gate.step(now - t_prev, now)
                    self.cmd = {s: v.copy() for s, v in cmd.items()}
                    dq = None if self.gate.dq is None else {s: v.copy() for s, v in self.gate.dq.items()}
                    engaged = self.gate.engaged
                t_prev = now
                if follows:
                    self.robot.set_engaged(engaged)           # chỉ phát lệnh khi đang engage
                if dq is not None:
                    self.robot.send(self.cmd, dq)
                else:
                    self.robot.send(self.cmd)
                if self.trace is not None:          # ghi 100 Hz: lệnh và góc đo (đo độ trễ, scripts/measure_lag.py)
                    meas = self.robot.read()
                    self.trace.append((now, {s: self.cmd[s].copy() for s in self.cmd},
                                       {s: np.asarray(meas[s], float).copy() for s in meas}))
                time.sleep(max(0.0, self.dt - (time.monotonic() - now)))
        except Exception as e:     # lỗi phần cứng: dừng vòng điều khiển, luồng chính sẽ thoát an toàn
            self.error = e
            self.running = False


def uncalibrated_free_wrists(pipe, gate):
    """Các tay robot có J5-J7 được phép cử động (giới hạn mềm khác [0, 0]) mà bàn tay chưa hiệu chuẩn."""
    out = []
    for s in pipe.robot_sides:
        free = bool(np.any(gate.hi[s][4:7] - gate.lo[s][4:7] > 1e-6))
        if free and not pipe.hand_calibrated[s]:
            out.append(s)
    return out


def park(robot, gate, rest, vel_deg_s, timeout=25.0):
    """Đưa hai tay về tư thế nghỉ với tốc độ thấp (chạy sau khi đã dừng luồng điều khiển)."""
    saved = gate.max_vel.copy()
    gate.max_vel = np.full(7, np.deg2rad(vel_deg_s))
    gate.engage(time.monotonic() - 10)     # bỏ qua pha tăng tốc
    if hasattr(robot, "returning"):
        robot.returning = True             # đang về: số đọc rác chỉ bị bỏ qua, không dừng giữa chừng
    if hasattr(robot, "set_engaged"):
        robot.set_engaged(True)            # backend ROS 2: được phát lệnh trong lúc về nghỉ
    try:
        t0 = t_prev = time.monotonic()
        while time.monotonic() - t0 < timeout:
            now = time.monotonic()
            tgt = {s: np.append(rest, np.nan) for s in gate.sides}
            gate.set_target(tgt, now)
            cmd = gate.step(now - t_prev, now)
            t_prev = now
            robot.send(cmd)
            if max(np.max(np.abs(cmd[s][:7] - rest)) for s in gate.sides) < np.deg2rad(1.0):
                break
            time.sleep(0.01)
    finally:
        if hasattr(robot, "returning"):
            robot.returning = False
        if hasattr(robot, "set_engaged"):
            robot.set_engaged(False)
        gate.max_vel = saved
        gate.disengage()
