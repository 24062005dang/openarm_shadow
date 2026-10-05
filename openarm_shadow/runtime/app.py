"""Vòng chạy chính: camera -> perception -> pipeline -> SafetyGate -> robot (sim hoặc OpenArm thật).

Ba luồng:
- luồng perception (runtime.worker): đọc camera + MediaPipe + fusion, luôn giữ kết quả mới nhất;
- luồng chính (hàm run): hiệu chuẩn tay, tự engage, retarget + lọc, đặt mục tiêu cho SafetyGate, vẽ, phím;
- luồng điều khiển (runtime.controller): chạy đều `control_hz`, bước SafetyGate (giới hạn vận tốc, dead-man,
  va chạm) rồi gửi lệnh xuống robot.

Phím: SPACE = engage / nhả thủ công (ly hợp) · c = hiệu chuẩn hướng bàn tay trung tính · g = hiệu chuẩn kẹp
      p = về tư thế nghỉ · q hoặc ESC = về tư thế nghỉ rồi thoát.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from ..cameras.multicam import MultiCameraSource
from ..cameras.sources import open_source
from ..fusion.multiview import MultiViewPerception
from ..perception.landmarker import Perception
from ..retarget.pipeline import ShadowPipeline
from ..robot import make_robot
from ..safety.gate import SafetyGate
from ..viz.draw import compose_view, draw_human, draw_robot, put_lines
from ..viz.hud import draw_ready_badge, status_lines
from .controller import Controller, park, uncalibrated_free_wrists
from .recorder import Recorder
from .session import AutoEngage
from .worker import PerceptionWorker


def open_perception(cfg, source):
    """Mở nguồn ảnh và bộ nhận diện. source "multi": nhiều camera (fusion.cameras), hợp nhất bằng triangulation;
    khác: một camera (RealSense, chỉ số webcam, file video, URL). -> (cap, perc, process, multi);
    process(sample) -> (Frame, extra) chạy trong luồng perception."""
    if str(source).lower() == "multi":
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

        def process(sample):
            fr = perc.process(sample)
            return fr, (list(perc.view_frames), perc.last_world)
        return cap, perc, process, True
    cap = open_source(source, cfg)
    perc = Perception(cfg["models"]["pose"], cfg["models"]["hand"], min_conf=cfg["models"]["min_conf"],
                      delegate=cfg["models"].get("delegate", "cpu"),
                      depth_cfg=cfg["camera"].get("realsense"), orientation_cfg=cfg.get("orientation"),
                      max_people=cfg["models"].get("pose_max_people", 2),
                      lock_dist=cfg["models"].get("pose_lock_dist", 1.0),
                      lock_keep_frames=cfg["models"].get("pose_lock_keep_frames", 15))

    def process(sample):
        return perc.process(sample.bgr, depth_m=sample.depth_m, depth_intrinsics=sample.intrinsics), None
    return cap, perc, process, False


def run(cfg, source, robot_kind="sim", record=None, show=True, dry_run=False):
    """dry_run (chỉ với robot_kind="openarm"): đọc góc robot thật, KHÔNG bật motor. Lệnh đi vào robot mô phỏng;
    hình vẽ có thêm nét xanh lá = tư thế đo từ robot thật. Dùng để kiểm tra can0/can1 và chiều từng khớp
    bằng cách cầm tay robot di chuyển, trước khi chạy thật."""
    cap, perc, process, multi = open_perception(cfg, source)
    robot = real = ctl = gate = worker = rec = None
    # Mọi thứ sau khi mở camera nằm trong try: lỗi ở bất kỳ bước nào (kể cả ngay sau khi bật motor) vẫn
    # về tư thế nghỉ và tắt motor, đóng camera.
    try:
        pipe = ShadowPipeline(cfg)
        if robot_kind == "openarm" and dry_run:
            real = make_robot("openarm", cfg, pipe.robot_sides)
            q_meas = real.connect()
            from ..robot.sim import SimRobot
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

        rec = Recorder(pipe.robot_sides) if record else None
        trace = rec.trace if rec else None
        ctl = Controller(robot, gate, cfg["robot"]["control_hz"])
        ctl.trace = trace
        ctl.start()
        rest = np.deg2rad(np.asarray(cfg["robot"]["rest_pose_deg"], float))
        fps_t, fps = time.monotonic(), 0.0
        auto = AutoEngage.from_config(cfg, robot_kind)
        msg = auto.hint() + "SPACE: dung/chay thu cong | c: calib lai | g: calib kep | p: ve nghi | q: thoat"
        if real is not None:
            msg = "DRY RUN: motor TAT. Xanh la = robot that. " + msg
        disp = cfg.get("display", {}) or {}
        display_period = 1.0 / max(0.1, float(disp.get("update_hz", 30.0)))
        last_display_t = -float("inf")
        compact_display = str(disp.get("mode", "full")).lower() == "compact"
        if show:
            cv2.namedWindow("openarm_shadow", cv2.WINDOW_NORMAL)   # kéo giãn được cửa sổ
        worker = PerceptionWorker(cap, process)
        worker.start()
        while ctl.running:
            item = worker.get()
            if item is None:
                print("DỪNG: mất khung camera.", worker.error)
                break
            sample, fr, extra = item
            with ctl.lock:
                engaged = gate.engaged
            auto_done = [] if engaged else pipe.auto_calibrate_hand_neutral(fr)
            if auto_done:
                print("Tự động hiệu chuẩn tay trung tính cho:", auto_done)
            ready_live = all(pipe.hand_calibrated[s] and pipe.calib_ready_now[s] for s in pipe.robot_sides)
            now = time.monotonic()
            if auto.update(ready_live, engaged, now):
                pipe.seed(robot.read())
                with ctl.lock:
                    gate.engage(now)
                engaged = True
                kind_text = "Robot thật" if robot_kind == "openarm" else "Simulation"
                print(kind_text, "tự đồng bộ sau khi READY liên tục đủ", auto.hold_s, "giây")
            targets = pipe.step(fr)
            for s, res in pipe.grip_calibration_results():
                if isinstance(res, tuple):
                    print(f"Kẹp {s}: pinch_ratio {res[0]:.2f}, open_ratio {res[1]:.2f} (đang dùng; muốn giữ cho lần "
                          f"sau thì ghi vào grip: trong config)")
                else:
                    print(f"Kẹp {s}: hiệu chuẩn không được - {res}. Bấm g thử lại.")
            with ctl.lock:
                gate.set_target(targets, time.monotonic(), fresh=pipe.fresh, t_frame=fr.t, held=pipe.held)
                cmd = {s: v.copy() for s, v in gate.cmd.items()}
                status = gate.status
            if rec:
                rec.add(pipe, fr, targets, cmd)
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - fps_t, 1e-3)
            fps_t = now
            if not show:
                continue
            if now - last_display_t >= display_period:
                last_display_t = now
                if multi:
                    cam = perc.draw(sample, fr, height=int(disp.get("camera_height", 360)),
                                    view_frames=extra[0], world=extra[1])
                else:
                    cam = draw_human(sample.bgr.copy(), fr)
                    if cfg["camera"]["mirror_display"]:
                        cam = cv2.flip(cam, 1)
                draw_ready_badge(cam, ready_live, engaged, auto.countdown)
                lines = status_lines([f"{fps:4.1f} fps | {status}", msg], pipe, fr, targets, cmd, multi,
                                     not multi and sample.depth_m is not None, compact_display)
                q_real = None
                if real is not None:
                    q_real = real.poll()
                    for s in pipe.robot_sides:
                        lines.append(f"{s} that (URDF, do): " +
                                     " ".join(f"{v:5.0f}" for v in np.rad2deg(q_real[s][:7])))
                elif robot_kind == "openarm":
                    q_real = robot.read()
                put_lines(cam, lines)
                title = "lenh (dam) / muc tieu (mo)" + (" / do that (xanh la)" if q_real else "")
                view = compose_view(cam, lambda size, zoom: draw_robot(pipe.kins, cmd, size=size, q_target=targets,
                                                                       q_meas=q_real, title=title,
                                                                       zoom_to_arms=zoom),
                                    layout=disp.get("robot_layout", "auto"),
                                    robot_height=disp.get("robot_height", 480))
                cv2.imshow("openarm_shadow", view)
            # Bắt phím mỗi khung perception, kể cả khi chỉ vẽ UI 8–10 Hz.
            k = cv2.waitKey(1) & 0xFF
            if k == ord(" "):
                auto.manual()
                with ctl.lock:
                    if gate.engaged:
                        gate.disengage()
                    elif blocked := uncalibrated_free_wrists(pipe, gate):
                        # Cổ tay J5-J7 được phép cử động mà chưa hiệu chuẩn tay: hướng trung tính mặc định có
                        # thể lệch tới 180° -> cổ tay chạy thẳng tới giới hạn khi engage. Không cho engage.
                        print("CHƯA ENGAGE: chưa hiệu chuẩn bàn tay cho", blocked,
                              "- thả tay xuôi, xoè bàn tay, lòng bàn tay nhìn camera, đứng yên tới khi READY.")
                    else:
                        pipe.seed(robot.read())
                        gate.engage(time.monotonic())
            elif k == ord("c"):
                if gate.engaged:
                    # Đổi offset cổ tay lúc đang bám làm lệnh J5-J7 nhảy: chỉ cho khi đã nhả (SPACE).
                    print("Nhả robot (SPACE) trước khi hiệu chuẩn lại bàn tay.")
                else:
                    print("Hiệu chuẩn tay trung tính cho:",
                          pipe.calibrate_hand_neutral(fr) or "không thấy bàn tay")
            elif k == ord("g"):
                pipe.start_grip_calibration(fr.t)
                print(f"Hiệu chuẩn kẹp {pipe.grip[pipe.robot_sides[0]].calib_s:.0f} s: chụm ngón cái-trỏ hết cỡ "
                      "rồi xoè hết cỡ, lặp lại vài lần.")
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
            if rec and rec.save(record):
                print("Đã lưu", record)
