"""Nối perception -> retarget -> lọc. Đầu ra: mục tiêu 8 phần tử cho mỗi tay robot
(7 góc URDF theo rad + độ mở kẹp 0..1; NaN = giữ nguyên khớp đó)."""
from __future__ import annotations

import numpy as np

from .filters import EMA, JointFilter
from .kinematics import ArmKinematics
from .perception import ArmObs, Frame
from .retarget import ArmRetargeter, mirror_rotation, mirror_vector


class ShadowPipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        mp = cfg["mapping"]
        self.robot_sides = list(mp["robot_arms"])
        self.mode = mp["mode"]                                  # "direct" hoặc "mirror"
        self.kins = {s: ArmKinematics(s) for s in self.robot_sides}
        rc = cfg["retarget"]
        self.rt = {s: ArmRetargeter(self.kins[s], rc["elbow_straight_deg"]) for s in self.robot_sides}
        fc = cfg["filter"]
        self.filt = {
            s: JointFilter(8, fc["min_cutoff"], fc["beta"], fc["deadband_deg"], fc["jump_deg"],
                           fc["jump_hold_s"], fc["min_conf"])
            for s in self.robot_sides
        }
        self.lm_ema = {s: EMA(fc["landmark_ema_alpha"]) for s in self.robot_sides}
        g = cfg["grip"]
        self.grip_pinch, self.grip_open = g["pinch_ratio"], g["open_ratio"]
        self.q_prev = {s: np.zeros(7) for s in self.robot_sides}
        self.last_info = {}

    # ------------------------------------------------------------------
    def human_side_for(self, robot_side):
        if self.mode == "direct":
            return robot_side                     # tay phải người -> tay phải robot
        return "left" if robot_side == "right" else "right"

    def _obs_for_robot(self, frame: Frame, robot_side) -> ArmObs:
        ob = frame.arms[self.human_side_for(robot_side)]
        if self.mode == "direct" or ob.s is None:
            return ob
        m = ArmObs(mirror_vector(ob.s), mirror_vector(ob.e), mirror_vector(ob.w),
                   mirror_rotation(ob.H), ob.grip, dict(ob.conf))
        return m

    def calibrate_hand_neutral(self, frame: Frame):
        """Người đứng/ngồi tư thế trung tính (tay thả xuôi, lòng bàn tay hướng vào thân) rồi gọi hàm này."""
        done = []
        for s in self.robot_sides:
            ob = self._obs_for_robot(frame, s)
            if ob.H is not None:
                self.rt[s].set_hand_neutral(ob.H)
                done.append(s)
        return done

    def seed(self, q_meas: dict):
        """Đặt nghiệm tham chiếu = tư thế robot hiện tại (để chọn nghiệm gần nhất)."""
        for s in self.robot_sides:
            self.q_prev[s] = np.asarray(q_meas[s][:7], float).copy()

    def step(self, frame: Frame):
        targets = {}
        for s in self.robot_sides:
            ob = self._obs_for_robot(frame, s)
            fc = self.cfg["filter"]
            c_up, c_fo, c_ha = ob.conf["upper"], ob.conf["fore"], ob.conf["hand"]
            ok_up, ok_fo, ok_ha = c_up >= fc["min_conf"], c_fo >= fc["min_conf"], c_ha >= fc["min_conf"]
            u = l = H = None
            if ob.s is not None and ok_up:
                pts = self.lm_ema[s](np.stack([ob.s, ob.e, ob.w]))
                s_, e_, w_ = pts
                u = e_ - s_
                if ok_fo:
                    l = w_ - e_
            if ok_ha and ob.H is not None:
                H = ob.H
            q, info = self.rt[s].solve(u, l, H, self.q_prev[s])
            self.q_prev[s] = q
            self.last_info[s] = info
            grip = np.nan
            if ob.grip is not None and ok_ha:
                grip = np.clip((ob.grip - self.grip_pinch) / (self.grip_open - self.grip_pinch), 0, 1)
            raw = np.append(q, grip)
            conf = np.array([c_up, c_up, c_fo, c_fo, c_ha, c_ha, c_ha, c_ha])
            if u is None:
                conf[:2] = 0
            if l is None:
                conf[2:4] = 0
            if H is None:
                conf[4:7] = 0
            if info.elbow_straight:
                conf[2] = 0          # J3 không xác định khi tay thẳng -> giữ
            out, _ = self.filt[s](raw, conf, frame.t)
            targets[s] = out
        return targets
