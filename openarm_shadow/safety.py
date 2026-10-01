"""Lớp an toàn đứng giữa pipeline và robot. Mọi lệnh tới robot (thật hay mô phỏng) đều đi qua đây.

- Giới hạn khớp mềm (hẹp hơn giới hạn URDF), giới hạn vận tốc từng khớp.
- Ly hợp (engage/disengage): chỉ bám người khi đã engage; mỗi lần engage tốc độ tăng dần (smoothstep).
- Dead-man: không có mục tiêu mới quá `deadman_s` giây -> đứng yên tại chỗ.
- Chống hai tay va nhau: mỗi đoạn tay là một capsule; bước nào làm khoảng cách xuống dưới
  ngưỡng và gần hơn trước thì bị bỏ (robot dừng tại chỗ, không nhảy).
Giá trị lệnh: 7 góc khớp theo URDF (rad) + phần tử thứ 8 = độ mở kẹp [0..1].
"""
from __future__ import annotations

import itertools

import numpy as np

from .geometry import seg_seg_distance
from .kinematics import ArmKinematics


def smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


class SafetyGate:
    def __init__(self, kins: dict[str, ArmKinematics], cfg: dict):
        self.kins = kins
        self.sides = list(kins)
        self.max_vel = np.deg2rad(np.asarray(cfg["max_vel_deg_s"], float))
        self.grip_vel = float(cfg.get("grip_vel_per_s", 1.5))
        self.deadman_s = float(cfg["deadman_s"])
        self.blend_s = float(cfg["engage_blend_s"])
        self.lo, self.hi = {}, {}
        for s, kin in kins.items():
            lim = np.deg2rad(np.asarray(cfg["soft_limits_deg"][s], float))
            self.lo[s] = np.maximum(lim[:, 0], kin.lower)
            self.hi[s] = np.minimum(lim[:, 1], kin.upper)
        col = cfg.get("self_collision", {})
        self.col_on = bool(col.get("enabled", True)) and len(self.sides) == 2
        self.col_margin = float(col.get("margin_m", 0.02))
        r = col.get("radius_m", {})
        self.radius = [r.get("upper", 0.05), r.get("fore", 0.045), r.get("hand", 0.05)]
        self.cmd = None
        self.target = None
        self.t_target = -np.inf
        self.engaged = False
        self.t_engage = 0.0
        self.status = "idle"
        # Khớp vừa đứng yên (bộ lọc giữ: mất tay, hướng tay chưa chắc) lâu hơn resume_after_s rồi chạy lại: tăng tốc
        # mềm riêng khớp đó trong resume_blend_s (không lao nhanh tới mục tiêu mới có thể đã ở xa).
        self.resume_after_s = float(cfg.get("resume_after_s", 0.3))
        self.resume_blend_s = float(cfg.get("resume_blend_s", 1.0))
        self._held_since = {s: np.full(7, np.nan) for s in self.sides}
        self.t_resume = {s: np.full(7, -np.inf) for s in self.sides}
        # Bám theo vận tốc (tuỳ chọn, mặc định tắt): xem _step_tracking.
        vt = cfg.get("velocity_tracking", {}) or {}
        self.vt_on = bool(vt.get("enabled", False))
        self.vt_gain = float(vt.get("gain", 8.0))
        self.vt_ext = float(vt.get("extrapolate_s", 0.07))
        self.vt_ff = bool(vt.get("feedforward", True))
        self.vt_alpha = float(vt.get("velocity_smoothing", 0.5))
        self.vt_min_step = np.deg2rad(float(vt.get("min_step_deg", 1.0)))
        self.max_acc = np.deg2rad(np.asarray(vt.get("max_acc_deg_s2", [300, 300, 400, 400, 1500, 1500, 1500]),
                                             float))
        self.v = {s: np.zeros(7) for s in self.sides}          # vận tốc lệnh hiện tại (rad/s, góc URDF)
        self.v_tgt = {s: np.zeros(7) for s in self.sides}      # vận tốc ước lượng của mục tiêu
        self._prev_tgt = {s: None for s in self.sides}
        self.dq = None                                         # vận tốc gửi motor (feedforward) hoặc None

    # ------------------------------------------------------------------
    def reset(self, q_meas: dict):
        """Gọi khi kết nối robot: lệnh bắt đầu đúng bằng tư thế đo được (không giật)."""
        self.cmd = {s: np.asarray(q_meas[s], float).copy() for s in self.sides}
        self.target = None
        self.engaged = False
        self._stop()

    def engage(self, now):
        self.engaged, self.t_engage = True, now

    def disengage(self):
        self.engaged = False

    def set_target(self, targets: dict, now, fresh=True, t_frame=None, held=None):
        """targets[side]: mảng 8 phần tử, NaN = giữ khớp đó. t_frame: thời điểm chụp khung camera tạo ra mục tiêu
        (dùng để ước lượng vận tốc cho velocity_tracking; nhận diện chạy ở luồng nền nên thời điểm nhận mục tiêu
        `now` dao động hơn thời điểm chụp). Mặc định = now.

        fresh=False: mục tiêu chỉ là giá trị cũ được bộ lọc giữ lại (không thấy người/tay): không làm mới đồng hồ
        dead-man, để quá deadman_s robot đứng yên. Có mục tiêu mới lại sau dead-man: tăng tốc mềm lại từ đầu
        (không lao nhanh tới mục tiêu có thể đã ở xa)."""
        self.target = {s: np.asarray(v, float).copy() for s, v in targets.items() if s in self.sides}
        self._track_held(held if held is not None else ({s: np.ones(8, bool) for s in self.sides} if not fresh
                                                         else None), now)
        if not fresh:
            if self.vt_on:            # mất người/tay: không ngoại suy tiếp theo vận tốc cũ
                for s in self.sides:
                    self.v_tgt[s][:] = 0.0
                    self._prev_tgt[s] = None
            return
        if self.vt_on:
            tf = now if t_frame is None else float(t_frame)
            for s, g in self.target.items():
                prev = self._prev_tgt[s]
                v = np.zeros(7)
                if prev is not None and 0.005 < tf - prev[1] < 0.3:
                    d = g[:7] - prev[0][:7]
                    ok = np.isfinite(d) & (np.abs(d) >= self.vt_min_step)   # rung nhỏ giữa 2 khung: không ngoại suy
                    v[ok] = np.clip(d[ok] / (tf - prev[1]), -self.max_vel[ok], self.max_vel[ok])
                # làm mượt vận tốc ước lượng (hiệu 2 khung camera rất nhiễu ở 15 fps)
                self.v_tgt[s] = self.vt_alpha * v + (1 - self.vt_alpha) * self.v_tgt[s]
                self._prev_tgt[s] = (g.copy(), tf)
        if self.engaged and now - self.t_target > 2 * self.deadman_s:
            self.t_engage = now          # mất mục tiêu thật sự (không chỉ 1 khung chậm): tăng tốc lại từ đầu
        self.t_target = now

    # ------------------------------------------------------------------
    def min_arm_distance(self, q: dict):
        if not self.col_on:
            return np.inf
        segs = {}
        for s in self.sides:
            k = self.kins[s].keypoints(q[s][:7])
            segs[s] = [(k["shoulder"], k["elbow"]), (k["elbow"], k["wrist"]), (k["wrist"], k["tool"])]
        a, b = self.sides
        d = np.inf
        for (i, sa), (j, sb) in itertools.product(enumerate(segs[a]), enumerate(segs[b])):
            if i == 0 and j == 0:
                continue          # hai cánh tay trên luôn cách xa nhau bởi thân
            dd = seg_seg_distance(*sa, *sb) - self.radius[i] - self.radius[j]
            d = min(d, dd)
        return d

    def _track_held(self, held, now):
        """held[side]: 8 cờ của bộ lọc (True = khớp đang giữ giá trị cũ). Khớp hết giữ sau > resume_after_s ->
        bắt đầu tăng tốc mềm riêng khớp đó."""
        if held is None:
            return
        for s in self.sides:
            h = np.asarray(held.get(s, np.zeros(8, bool)), bool)[:7]
            since = self._held_since[s]
            start = h & np.isnan(since)
            since[start] = now
            resumed = ~h & ~np.isnan(since)
            long_ = resumed & (now - np.nan_to_num(since, nan=now) > self.resume_after_s)
            self.t_resume[s][long_] = now
            since[resumed] = np.nan

    def joint_ramp(self, s, now):
        """Hệ số tốc độ riêng từng khớp (0,05..1) sau khi khớp đó chạy lại."""
        if self.resume_blend_s <= 0:
            return np.ones(7)
        return np.array([max(0.05, smoothstep((now - t) / self.resume_blend_s)) for t in self.t_resume[s]])

    def _stop(self):
        for s in self.sides:
            self.v[s][:] = 0.0
            self.v_tgt[s][:] = 0.0
            self._prev_tgt[s] = None
        self.dq = {s: np.zeros(7) for s in self.sides} if self.vt_on else None

    def _step_tracking(self, s, cur, goal, ramp, dt, now):
        """Bám theo vận tốc: lệnh chạy đều theo vận tốc ước lượng của mục tiêu (ngoại suy tối đa extrapolate_s
        giữa hai khung camera) + kéo về mục tiêu (gain), giới hạn vận tốc (max_vel x ramp) và gia tốc (max_acc).
        Thay cho kiểu cũ "lao tới mục tiêu ở tốc độ tối đa rồi đứng chờ khung sau" (vận tốc bật/tắt theo nhịp
        camera). Vận tốc lệnh được gửi kèm xuống motor (dq của MIT) để thành phần kd không còn hãm chuyển động."""
        q = cur[:7]
        tgt = self.target.get(s)
        v_tgt = self.v_tgt[s] if tgt is not None else np.zeros(7)
        v_tgt = np.where(np.isfinite(tgt[:7]), v_tgt, 0.0) if tgt is not None else v_tgt   # khớp đang giữ: 0
        age = max(now - self.t_target, 0.0)
        if age > self.vt_ext:
            v_tgt = np.zeros(7)       # mục tiêu cũ hơn extrapolate_s (khung camera đến trễ): thôi đẩy theo vận tốc
        goal_ext = np.clip(goal[:7] + v_tgt * min(age, self.vt_ext), self.lo[s], self.hi[s])
        vmax = self.max_vel * ramp * self.joint_ramp(s, now)
        v_des = np.clip(v_tgt + self.vt_gain * (goal_ext - q), -vmax, vmax)
        v = self.v[s] + np.clip(v_des - self.v[s], -self.max_acc * dt, self.max_acc * dt)
        v = np.clip(v, -vmax, vmax)
        q_new = q + v * dt
        hit = (q_new < self.lo[s]) | (q_new > self.hi[s])
        q_new = np.clip(q_new, self.lo[s], self.hi[s])
        v[hit] = 0.0
        self.v[s] = v
        grip = cur[7] + np.clip(goal[7] - cur[7], -self.grip_vel * ramp * dt, self.grip_vel * ramp * dt)
        return np.append(q_new, grip)

    def step(self, dt, now):
        if self.cmd is None:
            raise RuntimeError("SafetyGate.reset() chưa được gọi")
        if not self.engaged or self.target is None:
            self.status = "hold (chưa engage)" if not self.engaged else "hold (chưa có mục tiêu)"
            self._stop()
            return self.cmd
        if now - self.t_target > self.deadman_s:
            self.status = "hold (dead-man: mất mục tiêu)"
            self._stop()
            return self.cmd
        ramp = max(0.05, smoothstep((now - self.t_engage) / self.blend_s))
        new = {}
        for s in self.sides:
            cur = self.cmd[s]
            goal = self.target.get(s, cur)
            goal = np.where(np.isfinite(goal), goal, cur)
            goal[:7] = np.clip(goal[:7], self.lo[s], self.hi[s])
            goal[7] = np.clip(goal[7], 0.0, 1.0)
            if self.vt_on:
                new[s] = self._step_tracking(s, cur, goal, ramp, dt, now)
            else:
                vmax = np.append(self.max_vel * self.joint_ramp(s, now), self.grip_vel) * ramp * dt
                new[s] = cur + np.clip(goal - cur, -vmax, vmax)
        if self.vt_on:
            self.dq = {s: (self.v[s].copy() if self.vt_ff else np.zeros(7)) for s in self.sides}
        if self.col_on:
            d_new, d_cur = self.min_arm_distance(new), self.min_arm_distance(self.cmd)
            if d_new < self.col_margin and d_new < d_cur:
                # Bước đầy đủ làm hai tay xích lại quá gần. Không đứng im cả tay (dễ bị kẹt ở gần vùng va chạm),
                # mà chỉ cho đi từng khớp nào không làm khoảng cách giảm xuống dưới ngưỡng.
                part, d_part, moved = {k: v.copy() for k, v in self.cmd.items()}, d_cur, False
                for s in self.sides:
                    for i in np.flatnonzero(new[s] != part[s]):
                        trial = {k: v.copy() for k, v in part.items()}
                        trial[s][i] = new[s][i]
                        d_t = d_part if i == 7 else self.min_arm_distance(trial)   # kẹp không ảnh hưởng
                        if d_t >= self.col_margin or d_t >= d_part:
                            part, d_part, moved = trial, d_t, moved or i < 7
                self.cmd = part
                if self.vt_on:
                    self._stop()              # đang tránh va chạm: không gửi vận tốc, dừng mềm theo kp/kd
                self.status = (f"tránh va chạm: chỉ đi khớp an toàn, cách {d_part * 100:.1f} cm" if moved
                               else f"hold (sắp va chạm, cách {d_new * 100:.1f} cm)")
                return self.cmd
        self.cmd = new
        self.status = "follow" if ramp >= 1.0 else f"engage {ramp * 100:.0f}%"
        return self.cmd
