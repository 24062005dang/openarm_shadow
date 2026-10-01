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
from .multiview import MultiCameraSource, MultiViewPerception
from .sources import open_source
from .viz import draw_human, draw_robot, put_lines, side_by_side


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
        try:
            while self.running:
                now = time.monotonic()
                with self.lock:
                    cmd = self.gate.step(now - t_prev, now)
                    self.cmd = {s: v.copy() for s, v in cmd.items()}
                    dq = None if self.gate.dq is None else {s: v.copy() for s, v in self.gate.dq.items()}
                t_prev = now
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


def fusion_lines(fr, sides):
    """Dòng chẩn đoán fusion: số camera thấy vai/khuỷu/cổ tay, sai số chiếu lại, xung đột depth, bàn tay."""
    fi = fr.fusion or {}
    out = [f"fusion {fi.get('views', 0)} cam | lech khung {fi.get('skew_ms', 0.0):.0f} ms"
           + (f" | MAT KHUNG: {', '.join(fi['stale'])}" if fi.get("stale") else "")]
    names = {"right": (12, 14, 16), "left": (11, 13, 15)}
    for s in sides:
        pts = fi.get("points", {})
        if fi.get("body") == "front":
            out.append(f"{s}: vai/khuyu/co tay tu Pose camera 0")
            pts = None
        parts = []
        for tag, i in zip(("vai", "khuyu", "co tay"), names[s] if pts is not None else ()):
            p = pts.get(i)
            if p is None:
                parts.append(f"{tag} -")
                continue
            err = p.get("err_px", float("nan"))
            parts.append(f"{tag} {p['views']}cam" + (f" {err:.0f}px" if np.isfinite(err) else "") +
                         (" D" if p.get("depth") else "") + (" !" if p.get("conflict") else ""))
        if parts:
            out.append(f"{s}: " + " | ".join(parts))
        h = fi.get(f"hand_{s}")
        if h:
            extra = ""
            if h.get("fit") == "KABSCH" and np.isfinite(h.get("fit_mm", np.nan)):
                extra += f", khop long tay {h['fit_mm']:.0f}mm"
            elif h.get("fit") == "3PT":
                extra += ", dang hoc khuon long tay"
            if h.get("rejected"):
                extra += f", bo {h['rejected']} cam (xa co tay)"
            if h.get("bones_dropped"):
                extra += f", bo {h['bones_dropped']} diem (dot bat thuong)"
            if h.get("sources"):
                extra += " | nguon: " + "+".join(h["sources"])
            out.append(f"  ban tay {h['views']}cam {h['points']}/21 diem, nhin ro {h['quality']:.2f}, "
                       f"{h['mode']}{extra}")
    return out


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


class PerceptionWorker(threading.Thread):
    """Đọc camera + MediaPipe ở luồng riêng: luồng chính vẽ/điều khiển khung N trong lúc khung N+1 đang được nhận
    diện (trước đây làm nối tiếp nên vẽ chặn nhận diện). Luôn giữ kết quả MỚI NHẤT; luồng chính chậm thì bỏ khung cũ."""

    def __init__(self, cap, process):
        super().__init__(daemon=True, name="perception")
        self.cap, self.process = cap, process
        self.cond = threading.Condition()
        self.latest, self.seq, self.taken = None, 0, 0
        self.running, self.done, self.error = True, False, None

    def run(self):
        try:
            while self.running:
                ok, sample = self.cap.read()
                if not ok:
                    self.error = getattr(self.cap, "error", None) or "Nguồn video hết khung hoặc mất kết nối."
                    break
                item = (sample, *self.process(sample))
                with self.cond:
                    self.latest, self.seq = item, self.seq + 1
                    self.cond.notify_all()
        except BaseException as e:           # lỗi nhận diện: báo cho luồng chính, không chết im lặng
            self.error = f"{type(e).__name__}: {e}"
        finally:
            with self.cond:
                self.done = True
                self.cond.notify_all()

    def get(self, timeout=10.0):
        """Kết quả mới (sample, frame, extra) chưa lấy; None nếu luồng đã dừng (xem .error) hoặc quá timeout."""
        with self.cond:
            if not self.cond.wait_for(lambda: self.seq > self.taken or self.done, timeout):
                self.error = f"không có khung mới trong {timeout:.0f} s"
                return None
            if self.seq <= self.taken:
                return None
            self.taken = self.seq
            return self.latest

    def stop(self):
        self.running = False
        self.join(timeout=5.0)               # cap.read() có thể chờ tới 3 s


def run(cfg, source, robot_kind="sim", record=None, show=True, dry_run=False):
    """dry_run (chỉ với robot_kind="openarm"): đọc góc robot thật, KHÔNG bật motor. Lệnh đi vào robot mô phỏng;
    hình vẽ có thêm nét xanh lá = tư thế đo từ robot thật. Dùng để kiểm tra can0/can1 và chiều từng khớp
    bằng cách cầm tay robot di chuyển, trước khi chạy thật."""
    multi = str(source).lower() == "multi"
    if multi:
        # Nhiều camera (fusion.cameras): mỗi camera MediaPipe riêng, hợp nhất bằng triangulation (multiview.py).
        cap = MultiCameraSource(cfg)
        ok, first = cap.read(timeout=10.0)     # RealSense vừa mở có lúc cần > 3 s mới ra khung đầu
        if not ok:
            cap.close()
            raise SystemExit("Không đọc được khung từ đủ các camera trong fusion.cameras.\n" + (cap.error or ""))
        try:
            perc = MultiViewPerception.from_config(cfg, first)
        except BaseException:
            cap.close()
            raise
    else:
        cap = open_source(source, cfg)
        perc = Perception(cfg["models"]["pose"], cfg["models"]["hand"], min_conf=cfg["models"]["min_conf"],
                          delegate=cfg["models"].get("delegate", "cpu"),
                          depth_cfg=cfg["camera"].get("realsense"), orientation_cfg=cfg.get("orientation"))
    robot = real = ctl = gate = worker = trace = None
    log = {"t": []}
    # Mọi thứ sau khi mở camera nằm trong try: lỗi ở bất kỳ bước nào (kể cả ngay sau khi bật motor) vẫn
    # về tư thế nghỉ và tắt motor, đóng camera.
    try:
        pipe = ShadowPipeline(cfg)
        if robot_kind == "openarm" and dry_run:
            real = make_robot("openarm", cfg, pipe.robot_sides)
            q_meas = real.connect()
            from .robot.sim import SimRobot
            robot = SimRobot(pipe.robot_sides, q0=q_meas)
            robot_kind = "sim"
        else:
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
                raise SystemExit("Huỷ.")
            robot.enable()

        trace = [] if record else None
        ctl = Controller(robot, gate, cfg["robot"]["control_hz"])
        ctl.trace = trace
        ctl.start()
        rest = np.deg2rad(np.asarray(cfg["robot"]["rest_pose_deg"], float))
        log = {"t": [], **{f"target_{s}": [] for s in pipe.robot_sides}, **{f"cmd_{s}": [] for s in pipe.robot_sides}}
        fps_t, fps = time.monotonic(), 0.0
        auto_engage_s = float(cfg.get("calibration", {}).get("hand_auto", {}).get("auto_engage_sim_s", 3.0))
        ready_since = None
        auto_engage_used = False
        auto_countdown = None
        msg = ("GIU READY 3s: tu dong sync | SPACE: dung/chay thu cong | c: calib lai | q: thoat"
               if robot_kind == "sim" else
               "SPACE: engage | c: hieu chuan tay | p: ve nghi | q: thoat")
        if real is not None:
            msg = "DRY RUN: motor TAT. Xanh la = robot that. " + msg
        if show:
            cv2.namedWindow("openarm_shadow", cv2.WINDOW_NORMAL)   # kéo giãn được cửa sổ
        if multi:
            def process(sample):
                fr = perc.process(sample)
                return fr, (list(perc.view_frames), perc.last_world)
        else:
            def process(sample):
                return perc.process(sample.bgr, depth_m=sample.depth_m, depth_intrinsics=sample.intrinsics), None
        worker = PerceptionWorker(cap, process)
        worker.start()
        while ctl.running:
            item = worker.get()
            if item is None:
                print("DỪNG: mất khung camera.", worker.error)
                break
            sample, fr, extra = item
            frame_bgr = None if multi else sample.bgr
            with ctl.lock:
                engaged = gate.engaged
            auto_done = [] if engaged else pipe.auto_calibrate_hand_neutral(fr)
            if auto_done:
                print("Tự động hiệu chuẩn tay trung tính cho:", auto_done)
            ready_live = all(pipe.hand_calibrated[s] and pipe.calib_ready_now[s] for s in pipe.robot_sides)
            now = time.monotonic()
            if robot_kind == "sim" and not engaged and not auto_engage_used:
                if ready_live:
                    ready_since = now if ready_since is None else ready_since
                    auto_countdown = max(0.0, auto_engage_s - (now - ready_since))
                    if auto_countdown <= 0.0:
                        q_now = robot.read()
                        pipe.seed(q_now)
                        with ctl.lock:
                            gate.engage(now)
                        engaged = True
                        auto_engage_used = True
                        auto_countdown = None
                        print("Simulation tự đồng bộ sau khi READY đủ", auto_engage_s, "giây")
                else:
                    ready_since = None
                    auto_countdown = None
            targets = pipe.step(fr)
            with ctl.lock:
                gate.set_target(targets, time.monotonic(), fresh=pipe.fresh, t_frame=fr.t)
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
                if multi:
                    cam = perc.draw(sample, fr, view_frames=extra[0], world=extra[1])
                else:
                    cam = draw_human(frame_bgr.copy(), fr)
                    if cfg["camera"]["mirror_display"]:
                        cam = cv2.flip(cam, 1)
                ready = ready_live
                cx, cy = cam.shape[1] - 28, 28
                if ready and not engaged:
                    cv2.circle(cam, (cx, cy), 15, (0, 220, 0), -1)
                    ready_text = (f"AUTO SYNC {auto_countdown:.1f}s" if auto_countdown is not None
                                  else "READY")
                    cv2.putText(cam, ready_text, (max(8, cx - 175), cy + 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                elif ready:
                    cv2.circle(cam, (cx, cy), 15, (255, 200, 0), -1)
                    cv2.putText(cam, "FOLLOW", (max(8, cx - 90), cy + 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)
                else:
                    cv2.circle(cam, (cx, cy), 15, (0, 180, 255), 2)
                    cv2.putText(cam, "CALIB: ARM DOWN + PALM TO CAM", (max(8, cx - 285), cy + 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 2)
                lines = [f"{fps:4.1f} fps | {status}", msg]
                if multi:
                    lines += fusion_lines(fr, pipe.robot_sides)
                elif sample.depth_m is not None:
                    ds = " ".join(f"{s}:{fr.depth_used.get(s, 0)}/3" for s in pipe.robot_sides)
                    lines.append("D455 depth vai/khuyu/co tay " + ds + " (3/3 = dang dung depth)")
                    for human_side, di in fr.hand_depth.items():
                        lines.append(f"{human_side} hand: {di['mode']} depth {di['direct']}/21 "
                                     f"fused {di['fused']}/21 conf {di['confidence']:.2f}")
                        rms = di.get("plane_rms_m", float("inf"))
                        rms_text = f"{1000*rms:.1f}mm" if np.isfinite(rms) else "--"
                        lines.append(f"  orient {di.get('orientation', 'NONE')} open {di.get('open_fingers', 0)}/4 "
                                     f"plane {di.get('plane_inliers', 0)} rms {rms_text}")
                cs = " ".join(f"{s}:OK" if pipe.hand_calibrated[s] else
                              f"{s}:{100 * pipe.calib_progress[s]:.0f}% [{pipe.calib_hint[s]}]"
                              for s in pipe.robot_sides)
                lines.append("Auto calib tay: " + cs)
                for s in pipe.robot_sides:
                    inf = pipe.last_info.get(s)
                    if inf is not None:
                        lines.append(f"{s}: err u {inf.err_upper_deg:5.1f} l {inf.err_fore_deg:5.1f} "
                                     f"tay {inf.err_hand_deg:5.1f} deg" + (" [thang]" if inf.elbow_straight else ""))
                    if s in targets and np.all(np.isfinite(targets[s][4:7])):
                        wt = np.rad2deg(targets[s][4:7])
                        wc = np.rad2deg(cmd[s][4:7])
                        lines.append(f"{s} J5-7 target {wt[0]:5.1f} {wt[1]:5.1f} {wt[2]:5.1f} | "
                                     f"cmd {wc[0]:5.1f} {wc[1]:5.1f} {wc[2]:5.1f}")
                q_real = None
                if real is not None:
                    q_real = real.poll()
                    for s in pipe.robot_sides:
                        lines.append(f"{s} that (URDF, do): " +
                                     " ".join(f"{v:5.0f}" for v in np.rad2deg(q_real[s][:7])))
                elif robot_kind == "openarm":
                    q_real = robot.read()
                put_lines(cam, lines)
                rob = draw_robot(pipe.kins, cmd, q_target=targets, q_meas=q_real,
                                 title="lenh (dam) / muc tieu (mo)" + (" / do that (xanh la)" if q_real else ""))
                cv2.imshow("openarm_shadow", side_by_side(cam, rob))
                k = cv2.waitKey(1) & 0xFF
                if k == ord(" "):
                    auto_engage_used = True
                    ready_since = None
                    with ctl.lock:
                        if gate.engaged:
                            gate.disengage()
                        elif blocked := uncalibrated_free_wrists(pipe, gate):
                            # Cổ tay J5-J7 được phép cử động mà chưa hiệu chuẩn tay: hướng trung tính mặc định có
                            # thể lệch tới 180° -> cổ tay chạy thẳng tới giới hạn khi engage. Không cho engage.
                            print("CHƯA ENGAGE: chưa hiệu chuẩn bàn tay cho", blocked,
                                  "- thả tay xuôi, xoè bàn tay, lòng bàn tay nhìn camera, đứng yên tới khi READY.")
                        else:
                            q_now = robot.read()
                            pipe.seed(q_now)
                            gate.engage(time.monotonic())
                elif k == ord("c"):
                    if gate.engaged:
                        # Đổi offset cổ tay lúc đang bám làm lệnh J5-J7 nhảy: chỉ cho khi đã nhả (SPACE).
                        print("Nhả robot (SPACE) trước khi hiệu chuẩn lại bàn tay.")
                    else:
                        print("Hiệu chuẩn tay trung tính cho:",
                              pipe.calibrate_hand_neutral(fr) or "không thấy bàn tay")
                elif k == ord("p"):
                    with ctl.lock:
                        gate.disengage()
                    ctl.running = False
                    ctl.join()
                    park(robot, gate, rest, cfg["robot"]["park_vel_deg_s"])
                    ctl = Controller(robot, gate, cfg["robot"]["control_hz"])
                    ctl.trace = trace
                    ctl.start()
                elif k in (ord("q"), 27):
                    print("Thoát (phím q/Esc).")
                    break
    finally:
        ctl_error = None
        if ctl is not None:
            ctl.running = False
            ctl.join(timeout=1.0)
            ctl_error = ctl.error
            if ctl_error:
                print("LỖI vòng điều khiển:", ctl_error)
        try:
            if (robot_kind == "openarm" and robot is not None and getattr(robot, "enabled", False)
                    and ctl_error is None):
                print("Về tư thế nghỉ...")
                park(robot, gate, np.deg2rad(np.asarray(cfg["robot"]["rest_pose_deg"], float)),
                     cfg["robot"]["park_vel_deg_s"])
        finally:
            if worker is not None:
                worker.stop()                # dừng nhận diện trước khi đóng model/camera nó đang dùng
            if robot is not None:
                robot.close()
            if real is not None:
                real.close()
            perc.close()
            cap.close()
            cv2.destroyAllWindows()
            if record and log["t"]:
                if trace:
                    log["ctl_t"] = [x[0] for x in trace]
                    for s in trace[0][1]:
                        log[f"ctl_cmd_{s}"] = [x[1][s] for x in trace]
                        log[f"ctl_meas_{s}"] = [x[2][s] for x in trace]
                np.savez(record, **{k: np.asarray(v) for k, v in log.items()})
                print("Đã lưu", record)
