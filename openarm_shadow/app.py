"""Vòng chạy chính: camera -> pipeline -> SafetyGate -> robot (sim hoặc OpenArm thật).

Hai luồng:
- luồng chính: đọc camera, MediaPipe, retarget, lọc, đặt mục tiêu cho SafetyGate, vẽ.
- luồng điều khiển: chạy đều `control_hz`, bước SafetyGate (giới hạn vận tốc, dead-man, va chạm)
  rồi gửi lệnh xuống robot.

Phím: SPACE = engage / nhả (ly hợp) · c = hiệu chuẩn hướng bàn tay trung tính · b = học lại khung thân
      (orientation.body_ref) · p = về tư thế nghỉ
      q hoặc ESC = về tư thế nghỉ rồi thoát.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from openarm_shadow.control.controller import Controller, park, uncalibrated_free_wrists
from openarm_shadow.display.overlay import fusion_lines
from openarm_shadow.vision.landmarks import ARM_IDX
from openarm_shadow.vision.worker import PerceptionWorker
from openarm_shadow.vision.perception import Perception
from openarm_shadow.mapping.pipeline import ShadowPipeline
from openarm_shadow.robot import make_robot
from openarm_shadow.safety.gate import SafetyGate
from openarm_shadow.camera.multi_source import MultiCameraSource
from openarm_shadow.fusion.multiview import MultiViewPerception
from openarm_shadow.camera.sources import open_source
from openarm_shadow.display.viz import compose_view, draw_human, draw_robot, put_lines












def fusion_row(fr, human_side):
    """Chẩn đoán hợp nhất vai/khuỷu/cổ tay cho --record: [số camera x3, sai số chiếu lại px x3, depth x3] (NaN nếu
    không có, vd chế độ front hoặc 1 camera)."""
    pts = (getattr(fr, "fusion", None) or {}).get("points", {})
    row = np.full(9, np.nan)
    for k, i in enumerate(ARM_IDX[human_side]):
        p = pts.get(i)
        if p:
            row[k], row[3 + k], row[6 + k] = p.get("views", np.nan), p.get("err_px", np.nan), p.get("depth", np.nan)
    return row


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
                          depth_cfg=cfg["camera"].get("realsense"), orientation_cfg=cfg.get("orientation"),
                          max_people=cfg["models"].get("pose_max_people", 2),
                          lock_dist=cfg["models"].get("pose_lock_dist", 1.0),
                          lock_keep_frames=cfg["models"].get("pose_lock_keep_frames", 15))
    robot = real = ctl = gate = worker = trace = None
    log = {"t": []}
    # Mọi thứ sau khi mở camera nằm trong try: lỗi ở bất kỳ bước nào (kể cả ngay sau khi bật motor) vẫn
    # về tư thế nghỉ và tắt motor, đóng camera.
    try:
        pipe = ShadowPipeline(cfg)
        if robot_kind == "openarm" and dry_run:
            real = make_robot("openarm", cfg, pipe.robot_sides)
            q_meas = real.connect()
            from openarm_shadow.robot.sim import SimRobot
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

        simulated = robot_kind in ("sim", "mujoco")   # không có motor thật: tự engage khi READY, không hỏi yes
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
        log = {"t": [], **{f"{k}_{s}": [] for k in ("target", "cmd", "conf", "fus") for s in pipe.robot_sides}}
        fps_t, fps = time.monotonic(), 0.0
        auto_engage_s = float(cfg.get("calibration", {}).get("hand_auto", {}).get("auto_engage_sim_s", 3.0))
        ready_since = None
        auto_engage_used = False
        auto_countdown = None
        msg = ("GIU READY 3s: tu dong sync | SPACE: dung/chay thu cong | c: calib lai | b: hoc lai than | g: calib kep"
               " | q: thoat"
               if simulated else
               "SPACE: engage | c: hieu chuan tay | b: hoc lai than | g: calib kep | p: ve nghi | q: thoat")
        if real is not None:
            msg = "DRY RUN: motor TAT. Xanh la = robot that. " + msg
        disp = cfg.get("display", {}) or {}
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
            # Tham chiếu thân (orientation.body_ref) phải học xong trước: hiệu chuẩn tay và retarget đều theo khung thân
            ready_live = perc.body_ready() and all(pipe.hand_calibrated[s] and pipe.calib_ready_now[s]
                                                   for s in pipe.robot_sides)
            now = time.monotonic()
            if simulated and not engaged and not auto_engage_used:
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
            if record:
                log["t"].append(fr.t)
                for s in pipe.robot_sides:
                    log[f"target_{s}"].append(targets[s])
                    log[f"cmd_{s}"].append(cmd[s])
                    log[f"conf_{s}"].append(pipe.conf[s])
                    log[f"fus_{s}"].append(fusion_row(fr, pipe.human_side_for(s)))
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - fps_t, 1e-3)
            fps_t = now
            if show:
                if multi:
                    cam = perc.draw(sample, fr, view_frames=extra[0], world=extra[1])
                else:
                    ref = getattr(perc, "body_ref", None)
                    est = bool(ref is not None and ref.ready and ref.weights.get("R", 1.0) < 0.5)
                    cam = draw_human(frame_bgr.copy(), fr, torso_estimated=est)
                    if cfg["camera"]["mirror_display"]:
                        cam = cv2.flip(cam, 1)
                    if est:                          # sau khi lật: chữ không bị ngược
                        put_lines(cam, ["THAN: dung huong tham chieu (bi che)"], org=(10, cam.shape[0] - 40),
                                  color=(0, 140, 255))
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
                for s in pipe.robot_sides:
                    gm = pipe.grip[s]
                    rem = gm.calib_remaining(fr.t)
                    gl = (f"{s} kep: r {gm.r:4.2f} -> {gm.value:4.2f} (chum {gm.pinch:.2f} / xoe {gm.open:.2f})"
                          if np.isfinite(gm.r) else f"{s} kep: khong thay ngon cai/tro")
                    if rem is not None:
                        gl += f" | CALIB KEP {rem:3.1f}s: chum het co roi xoe het co"
                    lines.append(gl)
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
                lines.append(perc.body_status())
                for s in pipe.robot_sides:
                    inf = pipe.last_info.get(s)
                    if inf is not None:
                        lines.append(f"{s}: err u {inf.err_upper_deg:5.1f} l {inf.err_fore_deg:5.1f} "
                                     f"tay {inf.err_hand_deg:5.1f} deg" + (" [thang]" if inf.elbow_straight else ""))
                    ai = pipe.arm_shape_info.get(s)
                    if ai is not None:
                        # Lọc khung xương: độ dài đoạn tay khung này (cm) x tỉ lệ so với độ dài đã học; BO = khớp giữ
                        def seg(name, k):
                            r = ai["ratio"][k]
                            return (f"{name} {100 * ai['len'][k]:.0f}cm" + (f" x{r:.2f}" if np.isfinite(r) else
                                    " (dang hoc)") + ("" if ai["ok"][k] else " BO"))
                        lines.append(f"{s} xuong: " + seg("tren", 0) + " | " + seg("cang", 1))
                    if s in targets:
                        at, ac = np.rad2deg(targets[s][:4]), np.rad2deg(cmd[s][:4])
                        lines.append(f"{s} J1-4 target " + " ".join(f"{v:5.1f}" for v in at) + " | cmd " +
                                     " ".join(f"{v:5.1f}" for v in ac))
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
                elif robot_kind in ("openarm", "mujoco"):
                    q_real = robot.read()             # mujoco: góc thật trong mô phỏng (trễ / võng so với lệnh)
                put_lines(cam, lines)
                title = "lenh (dam) / muc tieu (mo)" + ((" / MuJoCo (xanh la)" if robot_kind == "mujoco" else
                                                          " / do that (xanh la)") if q_real else "")
                view = compose_view(cam, lambda size, zoom: draw_robot(pipe.kins, cmd, size=size, q_target=targets,
                                                                       q_meas=q_real, title=title,
                                                                       zoom_to_arms=zoom),
                                    layout=disp.get("robot_layout", "auto"),
                                    robot_height=disp.get("robot_height", 480))
                cv2.imshow("openarm_shadow", view)
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
                        elif not perc.body_ready():
                            print("CHƯA ENGAGE: chưa học tham chiếu thân -", perc.body_status(),
                                  "- đứng thẳng, thả tay xuôi cho camera thấy rõ vai và hông.")
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
                elif k == ord("b"):
                    if gate.engaged:
                        print("Nhả robot (SPACE) trước khi học lại khung thân.")
                    else:
                        # Hướng tay trung tính đã hiệu chuẩn theo khung thân cũ -> hiệu chuẩn lại cùng lúc (tự động
                        # khi thả tay xuôi, cùng tư thế dùng để học khung thân).
                        perc.relearn_body()
                        for s in pipe.robot_sides:
                            pipe.hand_calibrated[s] = False
                            pipe._reset_auto_calib(s)
                        print("Học lại khung thân + hiệu chuẩn tay: đứng thẳng, thả tay xuôi, xoè bàn tay, đứng yên.")
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
            if record and log["t"]:
                if trace:
                    log["ctl_t"] = [x[0] for x in trace]
                    for s in trace[0][1]:
                        log[f"ctl_cmd_{s}"] = [x[1][s] for x in trace]
                        log[f"ctl_meas_{s}"] = [x[2][s] for x in trace]
                np.savez(record, **{k: np.asarray(v) for k, v in log.items()})
                print("Đã lưu", record)
