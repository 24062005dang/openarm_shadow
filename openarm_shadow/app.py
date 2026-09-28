"""Vòng chạy chính: camera -> pipeline -> SafetyGate -> robot (sim hoặc OpenArm thật).

Hai luồng:
- luồng chính: đọc camera, MediaPipe, retarget, lọc, đặt mục tiêu cho SafetyGate, vẽ.
- luồng điều khiển: chạy đều `control_hz`, bước SafetyGate (giới hạn vận tốc, dead-man, va chạm)
  rồi gửi lệnh xuống robot.

Phím: SPACE = engage / nhả (ly hợp) · c = hiệu chuẩn hướng bàn tay trung tính · p = về tư thế nghỉ
      q hoặc ESC = về tư thế nghỉ rồi thoát.
"""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from .perception import Perception
from .pipeline import ShadowPipeline
from .robot import make_robot
from .safety import SafetyGate
from .viz import draw_human, draw_robot, put_lines, side_by_side


def open_source(src):
    cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
    if not cap.isOpened():
        raise SystemExit(f"Không mở được nguồn video: {src}")
    return cap


class Controller(threading.Thread):
    def __init__(self, robot, gate, hz):
        super().__init__(daemon=True)
        self.robot, self.gate, self.dt = robot, gate, 1.0 / hz
        self.lock = threading.Lock()
        self.running = True
        self.error = None
        self.cmd = None

    def run(self):
        t_prev = time.monotonic()
        try:
            while self.running:
                now = time.monotonic()
                with self.lock:
                    cmd = self.gate.step(now - t_prev, now)
                    self.cmd = {s: v.copy() for s, v in cmd.items()}
                t_prev = now
                self.robot.send(self.cmd)
                time.sleep(max(0.0, self.dt - (time.monotonic() - now)))
        except Exception as e:     # lỗi phần cứng: dừng vòng điều khiển, luồng chính sẽ thoát an toàn
            self.error = e
            self.running = False


def park(robot, gate, rest, vel_deg_s, timeout=25.0):
    """Đưa hai tay về tư thế nghỉ với tốc độ thấp (chạy sau khi đã dừng luồng điều khiển)."""
    saved = gate.max_vel.copy()
    gate.max_vel = np.full(7, np.deg2rad(vel_deg_s))
    gate.engage(time.monotonic() - 10)     # bỏ qua pha tăng tốc
    if hasattr(robot, "returning"):
        robot.returning = True             # đang về: số đọc rác chỉ bị bỏ qua, không dừng giữa chừng
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
        gate.max_vel = saved
        gate.disengage()


def run(cfg, source, robot_kind="sim", record=None, show=True):
    cap = open_source(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["camera"]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera"]["height"])
    perc = Perception(cfg["models"]["pose"], cfg["models"]["hand"], min_conf=cfg["models"]["min_conf"])
    pipe = ShadowPipeline(cfg)
    robot = make_robot(robot_kind, cfg, pipe.robot_sides)
    q_meas = robot.connect()
    print("Tư thế đo được (độ, URDF):")
    for s, q in q_meas.items():
        print(f"  {s:5s}", np.round(np.rad2deg(q[:7]), 1), " kẹp", round(float(q[7]), 2))
    gate = SafetyGate(pipe.kins, cfg["safety"])
    gate.reset(q_meas)
    pipe.seed(q_meas)

    if robot_kind == "openarm":
        print("\nROBOT THẬT. Kiểm tra: E-stop trong tay, không ai trong tầm với, tay đang thả xuôi.")
        if input("Gõ 'yes' để bật motor: ").strip().lower() != "yes":
            robot.close()
            raise SystemExit("Huỷ.")
        robot.enable()

    ctl = Controller(robot, gate, cfg["robot"]["control_hz"])
    ctl.start()
    rest = np.deg2rad(np.asarray(cfg["robot"]["rest_pose_deg"], float))
    log = {"t": [], **{f"target_{s}": [] for s in pipe.robot_sides}, **{f"cmd_{s}": [] for s in pipe.robot_sides}}
    fps_t, fps = time.monotonic(), 0.0
    msg = "SPACE: engage | c: hieu chuan tay | p: ve nghi | q: thoat"
    try:
        while ctl.running:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            fr = perc.process(frame_bgr)
            targets = pipe.step(fr)
            with ctl.lock:
                gate.set_target(targets, time.monotonic())
                cmd = {s: v.copy() for s, v in gate.cmd.items()}
                status = gate.status
            if record:
                log["t"].append(fr.t)
                for s in pipe.robot_sides:
                    log[f"target_{s}"].append(targets[s])
                    log[f"cmd_{s}"].append(cmd[s])
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - fps_t, 1e-3)
            fps_t = now
            if show:
                cam = draw_human(frame_bgr.copy(), fr)
                if cfg["camera"]["mirror_display"]:
                    cam = cv2.flip(cam, 1)
                lines = [f"{fps:4.1f} fps | {status}", msg]
                for s in pipe.robot_sides:
                    inf = pipe.last_info.get(s)
                    if inf is not None:
                        lines.append(f"{s}: err u {inf.err_upper_deg:5.1f} l {inf.err_fore_deg:5.1f} "
                                     f"tay {inf.err_hand_deg:5.1f} deg" + (" [thang]" if inf.elbow_straight else ""))
                put_lines(cam, lines)
                rob = draw_robot(pipe.kins, cmd, q_target=targets, title="lenh (dam) / muc tieu (mo)")
                cv2.imshow("openarm_shadow", side_by_side(cam, rob))
                k = cv2.waitKey(1) & 0xFF
                if k == ord(" "):
                    with ctl.lock:
                        if gate.engaged:
                            gate.disengage()
                        else:
                            q_now = robot.read()
                            pipe.seed(q_now)
                            gate.engage(time.monotonic())
                elif k == ord("c"):
                    print("Hiệu chuẩn tay trung tính cho:", pipe.calibrate_hand_neutral(fr) or "không thấy bàn tay")
                elif k == ord("p"):
                    with ctl.lock:
                        gate.disengage()
                    ctl.running = False
                    ctl.join()
                    park(robot, gate, rest, cfg["robot"]["park_vel_deg_s"])
                    ctl = Controller(robot, gate, cfg["robot"]["control_hz"])
                    ctl.start()
                elif k in (ord("q"), 27):
                    break
    finally:
        ctl.running = False
        ctl.join(timeout=1.0)
        if ctl.error:
            print("LỖI vòng điều khiển:", ctl.error)
        try:
            if robot_kind == "openarm" and ctl.error is None:
                print("Về tư thế nghỉ...")
                park(robot, gate, rest, cfg["robot"]["park_vel_deg_s"])
        finally:
            robot.close()
            perc.close()
            cap.release()
            cv2.destroyAllWindows()
            if record and log["t"]:
                np.savez(record, **{k: np.asarray(v) for k, v in log.items()})
                print("Đã lưu", record)
