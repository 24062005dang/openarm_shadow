"""Nối perception -> retarget -> lọc. Đầu ra: mục tiêu 8 phần tử cho mỗi tay robot
(7 góc URDF theo rad + độ mở kẹp 0..1; NaN = giữ nguyên khớp đó)."""
from __future__ import annotations

import numpy as np

from openarm_shadow.core.geometry import angle_between, orthonormalize, unit
from openarm_shadow.mapping.arm import Arm, _SideView
from openarm_shadow.core.types import ArmObs, Frame
from openarm_shadow.mapping.retarget import mirror_rotation, mirror_vector


class ShadowPipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        mp = cfg["mapping"]
        self.robot_sides = list(mp["robot_arms"])
        self.mode = mp["mode"]                                  # "direct" hoặc "mirror"
        rc = cfg["retarget"]
        # Mỗi tay robot là một đối tượng Arm (cùng một lớp cho trái / phải, xem arm.py); dict theo tay bên dưới là
        # "view" trỏ vào Arm để mã và test cũ (pipe.held["right"], pipe.rt["left"]...) chạy tiếp.
        self.arms = {s: Arm(s, cfg, self.human_side_for(s), mirrored=self.mode != "direct") for s in self.robot_sides}
        for name in ("kins:kin", "rt", "filt", "lm_ema", "grip", "q_prev", "last_info", "held", "conf", "kf_rejected",
                     "arm_shape_info", "hand_calibrated", "calib_ready_now", "calib_progress", "calib_hint",
                     "_calib_samples", "_calib_start", "_calib_prev"):
            view, _, attr = name.partition(":")
            setattr(self, view, _SideView(self.arms, attr or view))
        # J3 (xoay cánh tay) càng gần thẳng càng kém xác định (trục J3 trùng trục J5 khi tay thẳng): độ tin cậy J3
        # tăng dần từ 0 ở elbow_straight_deg tới đủ ở elbow_j3_full_deg.
        self.j3_bend = (float(rc["elbow_straight_deg"]), float(rc.get("elbow_j3_full_deg", 30.0)))
        fc = cfg["filter"]
        kc = dict(fc.get("landmark_kalman") or {})
        self.lm_kf = _SideView(self.arms, "lm_kf") if kc.get("enabled", False) else None
        self.bone_fix = bool(kc.get("bone_length_fix", True))
        self.arm_shape = _SideView(self.arms, "arm_shape") if dict(fc.get("arm_shape") or {}).get("enabled", False) else None
        self.fresh = False       # khung vừa rồi có ít nhất 1 khớp nhận giá trị mới (không phải giữ) -> dead-man
        self.raw_targets = {s: np.full(8, np.nan) for s in self.robot_sides}   # nghiệm IK + kẹp trước bộ lọc (--record)
        cc = cfg.get("calibration", {}).get("hand_auto", {})
        self.auto_calib_enabled = bool(cc.get("enabled", True))
        self.auto_calib_hold_s = float(cc.get("hold_s", 0.6))
        self.auto_calib_min_samples = int(cc.get("min_samples", 5))
        self.auto_calib_motion = np.deg2rad(float(cc.get("max_motion_deg", 12)))
        self.auto_calib_min_open = int(cc.get("min_open_fingers", 4))
        self.auto_calib_upper_down = np.deg2rad(float(cc.get("max_upper_from_down_deg", 55)))
        self.auto_calib_fore_down = np.deg2rad(float(cc.get("max_fore_from_down_deg", 55)))
        self.auto_calib_palm_camera = np.deg2rad(float(cc.get("max_palm_from_camera_deg", 50)))

    # ------------------------------------------------------------------
    def human_side_for(self, robot_side):
        if self.mode == "direct":
            return robot_side                     # tay phải người -> tay phải robot
        return "left" if robot_side == "right" else "right"

    def _fix_bone_length(self, side, pts):
        """Khuỷu / cổ tay bị Kalman loại (đang dùng dự đoán): đặt lại đúng độ dài cánh tay trên / cẳng tay đã học
        (ArmShape), giữ hướng của dự đoán. Dự đoán thuần có thể co / giãn đoạn tay khi điểm bị khuất vài khung."""
        ref = self.arm_shape[side].ref()
        s_, e_, w_ = (np.asarray(p, float).copy() for p in pts)
        rej = self.kf_rejected[side]
        if np.isfinite(ref[0]) and rej[1] and np.linalg.norm(e_ - s_) > 1e-6:
            e_ = s_ + ref[0] * unit(e_ - s_)                         # cổ tay đo tốt thì giữ nguyên cổ tay
        if np.isfinite(ref[1]) and rej[2] and np.linalg.norm(w_ - e_) > 1e-6:
            w_ = e_ + ref[1] * unit(w_ - e_)
        return np.stack([s_, e_, w_])

    @staticmethod
    def _point_conf(ob, c_up, c_fo):
        """Độ tin cậy riêng vai, khuỷu, cổ tay cho Kalman điểm. Trước đây dùng một số chung cho cả 3 điểm (= của cánh
        tay trên khi cẳng tay nhìn kém) nên cổ tay sai vẫn được tin như điểm tốt. Nguồn không có conf["points"]
        (dữ liệu giả lập, file cũ): vai/khuỷu theo cánh tay trên, cổ tay theo cẳng tay."""
        pc = ob.conf.get("points")
        if pc is None:
            pc = (c_up, c_up, c_fo)
        return np.clip(np.asarray(pc, float), 0.0, 1.0)

    def _obs_for_robot(self, frame: Frame, robot_side) -> ArmObs:
        ob = frame.arms[self.human_side_for(robot_side)]
        if self.mode == "direct" or ob.s is None:
            return ob
        m = ArmObs(s=mirror_vector(ob.s), e=mirror_vector(ob.e), w=mirror_vector(ob.w),
                   H=mirror_rotation(ob.H), grip=ob.grip, conf=dict(ob.conf),
                   hand_points_cam=ob.hand_points_cam, hand_depth_valid=ob.hand_depth_valid,
                   hand_depth_confidence=ob.hand_depth_confidence, hand_depth_mode=ob.hand_depth_mode,
                   hand_R_cam=ob.hand_R_cam, hand_center_cam=ob.hand_center_cam,
                   hand_axes_px=ob.hand_axes_px, hand_open_fingers=ob.hand_open_fingers,
                   hand_orientation_mode=ob.hand_orientation_mode)
        return m

    def _neutral_reference(self, side, ob):
        """Giữ J1–J4 theo tư thế tay hiện tại; J5–J7 = góc robot ứng với tư thế hiệu chuẩn (lòng bàn tay nhìn camera),
        xem Arm.calib_wrist: hai ngón kẹp đóng / mở theo pháp tuyến lòng bàn tay."""
        u = unit(ob.e - ob.s)
        l = unit(ob.w - ob.e)
        q_ref, _ = self.rt[side].solve(u, l, None, self.q_prev[side])
        q_ref[4:7] = self.arms[side].calib_wrist
        return q_ref

    def calibrate_hand_neutral(self, frame: Frame):
        """Calib hướng tay tại tư thế hiện tại; J5–J7 của robot được xem là trung tính."""
        done = []
        for s in self.robot_sides:
            ob = self._obs_for_robot(frame, s)
            if ob.H is not None and ob.s is not None and ob.e is not None and ob.w is not None:
                self.rt[s].set_hand_neutral(ob.H, self._neutral_reference(s, ob))
                self.hand_calibrated[s] = True
                self.calib_progress[s] = 1.0
                self.calib_hint[s] = "READY"
                self._reset_auto_calib(s, keep_progress=True)
                done.append(s)
        return done

    def _reset_auto_calib(self, side, keep_progress=False):
        self.arms[side].reset_auto_calib(keep_progress)

    def _calib_pose_status(self, ob):
        down = np.array([0.0, 0.0, -1.0])
        toward_camera = np.array([0.0, 0.0, -1.0])
        valid = ob.H is not None and ob.s is not None and ob.e is not None and ob.w is not None
        hint = "dua tay vao khung"
        if valid and ob.conf["hand"] < self.cfg["filter"]["min_conf"]:
            valid, hint = False, "tay chua ro"
        if valid and ob.hand_open_fingers < self.auto_calib_min_open:
            valid, hint = False, "xoe ban tay"
        if valid and ob.hand_R_cam is None:
            valid, hint = False, "cho depth ban tay"
        u = None
        if valid:
            u, l = unit(ob.e - ob.s), unit(ob.w - ob.e)
            if angle_between(u, down) > self.auto_calib_upper_down:
                valid, hint = False, "tha canh tay xuong"
            elif angle_between(l, down) > self.auto_calib_fore_down:
                valid, hint = False, "tha co tay xuong"
            elif angle_between(ob.hand_R_cam[:, 2], toward_camera) > self.auto_calib_palm_camera:
                valid, hint = False, "long ban tay nhin camera"
        return valid, hint, u

    def auto_calibrate_hand_neutral(self, frame: Frame):
        """Tự calib khi tay buông xuống hơi dang, khuỷu thả lỏng, lòng bàn tay nhìn camera.

        Chỉ gọi khi SafetyGate chưa engage để thay offset cổ tay không làm lệnh nhảy trong lúc follow.
        """
        done = []
        if not self.auto_calib_enabled:
            return done
        for s in self.robot_sides:
            ob = self._obs_for_robot(frame, s)
            valid, hint, u = self._calib_pose_status(ob)
            self.calib_ready_now[s] = bool(valid)
            if self.hand_calibrated[s]:
                self.calib_hint[s] = "READY" if valid else hint
                continue
            if not valid:
                self.calib_hint[s] = hint
                self._reset_auto_calib(s)
                continue
            self.calib_hint[s] = "giu yen"
            prev = self._calib_prev[s]
            if prev is not None:
                prev_u, prev_H = prev
                dH = np.asarray(prev_H).T @ np.asarray(ob.H)
                hand_motion = np.arccos(np.clip((np.trace(dH) - 1) / 2, -1, 1))
                if angle_between(prev_u, u) > self.auto_calib_motion or hand_motion > self.auto_calib_motion:
                    self.calib_hint[s] = "giu yen"
                    self._reset_auto_calib(s)
            if self._calib_start[s] is None:
                self._calib_start[s] = frame.t
            self._calib_samples[s].append(np.asarray(ob.H).copy())
            self._calib_prev[s] = (u.copy(), np.asarray(ob.H).copy())
            elapsed = max(0.0, frame.t - self._calib_start[s])
            self.calib_progress[s] = min(1.0, elapsed / max(self.auto_calib_hold_s, 1e-3))
            if elapsed >= self.auto_calib_hold_s and len(self._calib_samples[s]) >= self.auto_calib_min_samples:
                H = orthonormalize(np.mean(self._calib_samples[s], axis=0))
                self.rt[s].set_hand_neutral(H, self._neutral_reference(s, ob))
                self.hand_calibrated[s] = True
                self.calib_hint[s] = "READY"
                self._reset_auto_calib(s, keep_progress=True)
                done.append(s)
        return done

    def start_grip_calibration(self, t):
        """Phím g: trong grip.calib_s giây chụm hết cỡ rồi xoè hết cỡ vài lần -> đặt lại pinch/open theo tay người."""
        for g in self.grip.values():
            g.start_calibration(t)

    def grip_calibration_results(self):
        """[(tay, (pinch, open) hoặc chuỗi lỗi)] của các lần hiệu chuẩn kẹp vừa xong (mỗi kết quả trả một lần)."""
        out = []
        for s, g in self.grip.items():
            if g.calib_result is not None:
                out.append((s, g.calib_result))
                g.calib_result = None
        return out

    def seed(self, q_meas: dict):
        """Đặt nghiệm tham chiếu = tư thế robot hiện tại (để chọn nghiệm gần nhất)."""
        for s in self.robot_sides:
            self.q_prev[s] = np.asarray(q_meas[s][:7], float).copy()

    def step(self, frame: Frame):
        targets, fresh = {}, False
        for s in self.robot_sides:
            ob = self._obs_for_robot(frame, s)
            fc = self.cfg["filter"]
            c_up, c_fo, c_ha = ob.conf["upper"], ob.conf["fore"], ob.conf["hand"]
            c_gr = ob.conf.get("grip", c_ha)        # fusion: kẹp có độ tin cậy riêng (không theo hướng bàn tay)
            ok_up, ok_fo, ok_ha = c_up >= fc["min_conf"], c_fo >= fc["min_conf"], c_ha >= fc["min_conf"]
            u = l = H = None
            if self.arm_shape is not None:
                self.arm_shape_info[s] = None
                if ob.s is not None and ok_up:
                    good_up, good_fo, self.arm_shape_info[s] = self.arm_shape[s](ob.s, ob.e, ob.w, frame.t,
                                                                                 (ok_up, ok_up and ok_fo))
                    if not (good_up and good_fo):
                        # Một đoạn sai độ dài thường do khuỷu sai (làm sai hướng CẢ hai đoạn, nhưng độ dài chỉ lộ ra ở
                        # một đoạn) -> giữ cả J1-J4, không phân biệt được khuỷu hay cổ tay sai.
                        c_up = c_fo = 0.0
                        ok_up = ok_fo = False
            if ob.s is not None and ok_up:
                raw_pts = np.stack([ob.s, ob.e, ob.w])
                if self.lm_kf is not None:
                    pts = self.lm_kf[s](raw_pts, frame.t, self._point_conf(ob, c_up, c_fo))
                    rej = self.lm_kf[s].rejected
                    self.kf_rejected[s] = np.zeros(3, bool) if rej is None else np.asarray(rej, bool).copy()
                    if self.bone_fix and self.arm_shape is not None and self.kf_rejected[s][1:].any():
                        pts = self._fix_bone_length(s, pts)
                else:
                    pts = self.lm_ema[s](raw_pts)
                s_, e_, w_ = pts
                u = e_ - s_
                if ok_fo:
                    l = w_ - e_
            if ok_ha and ob.H is not None:
                H = ob.H
            q, info = self.rt[s].solve(u, l, H, self.q_prev[s])
            self.q_prev[s] = q
            self.last_info[s] = info
            wrist = pts[2] if (ob.s is not None and ok_up) else ob.w     # tay đang di chuyển: kẹp khó nhả nhầm
            grip = self.grip[s](ob.grip if c_gr >= fc["min_conf"] else None, frame.t, wrist=wrist,
                                r_views=getattr(ob, "grip_views", None))
            raw = np.append(q, grip)
            self.raw_targets[s] = raw.copy()
            conf = np.array([c_up, c_up, c_fo, c_fo, c_ha, c_ha, c_ha, c_gr])
            if u is None:
                conf[:2] = 0
            if l is None:
                conf[2:4] = 0
            if H is None:
                conf[4:7] = 0
            if info.elbow_straight:
                conf[2] = 0          # J3 không xác định khi tay thẳng -> giữ
            elif np.isfinite(info.elbow_bend_deg):
                a, b = self.j3_bend
                conf[2] *= float(np.clip((info.elbow_bend_deg - a) / max(b - a, 1e-6), 0.0, 1.0))
            out, held = self.filt[s](raw, conf, frame.t)
            self.held[s] = held.copy()
            self.conf[s] = conf.copy()
            fresh = fresh or not bool(np.all(held[:7]))
            targets[s] = out
        self.fresh = fresh
        return targets
