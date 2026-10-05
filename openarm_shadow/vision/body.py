"""Khung thân người và tham chiếu thân lọc theo độ nhất quán (BodyRef)."""
from __future__ import annotations

import numpy as np

from openarm_shadow.core.geometry import make_frame, orthonormalize
from openarm_shadow.core.rotation import rotation_distance, slerp_rotation
from openarm_shadow.vision.landmarks import L_HIP, L_SH, R_HIP, R_SH


def body_frame(W, vis, min_hip_vis=0.5):
    """Khung thân từ điểm world của Pose. Thiếu hông (ngồi, bị bàn che) thì dùng 'lên' của camera."""
    if min(vis[L_HIP], vis[R_HIP]) >= min_hip_vis:
        bottom = 0.5 * (W[L_HIP] + W[R_HIP])
    else:
        bottom = 0.5 * (W[L_SH] + W[R_SH]) + np.array([0.0, 0.5, 0.0])   # y của MediaPipe hướng xuống
    return make_frame(W[L_SH], W[R_SH], bottom)


class BodyRef:
    """Tham chiếu thân có lọc theo ĐỘ NHẤT QUÁN (orientation.body_ref): chỉ ước lượng khi điểm đo mâu thuẫn.

    Người điều khiển đứng gần như yên, chỉ hai tay cử động. Khi đưa hai tay ra trước, tay che thân -> MediaPipe vẫn
    báo vai/hông "thấy rõ" nhưng đặt sai chỗ, nên các cơ chế dựa vào độ tin cậy (pose_hold, Kalman qua đoạn mất điểm)
    không bắt được. Ở đây so điểm đo với tham chiếu đã học:

    - lệch < tol[0]: dùng điểm đo, tham chiếu trôi chậm theo (follow) -> giữ chuyển động thật nhỏ (thở, nhún vai);
    - lệch > tol[1]: coi là bị che, dùng tham chiếu; ở giữa trộn tuyến tính (không đổi đột ngột);
    - mọi điểm cùng lệch nhưng khoảng cách giữa chúng giữ nguyên (khối cứng) liên tục adopt_s: người xoay / dịch
      thật -> nhận vị trí mới làm tham chiếu. Bị che thường chỉ vài điểm lệch và hình thân bị méo.

    Fusion triangulate (mọi điểm cùng hệ camera 0): lọc từng điểm vai/hông (gate_points), khung thân dựng lại từ điểm
    đã lọc. 1 camera (điểm lúc depth lúc MediaPipe world, không chung hệ): chỉ lọc HƯỚNG khung thân (gate_rotation).
    Học tham chiếu: learn_s giây liên tục thấy rõ vai (+ hông nếu require_hips) và đứng yên. Học lại: phím b, hoặc mất
    người lâu hơn relearn_lost_s."""

    def __init__(self, cfg=None):
        cfg = dict(cfg or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.learn_s = float(cfg.get("learn_s", 1.0))
        self.min_vis = float(cfg.get("min_vis", 0.6))
        self.require_hips = bool(cfg.get("require_hips", False))
        self.max_motion = np.deg2rad(float(cfg.get("max_motion_deg", 10.0)))
        self.max_shift_m = float(cfg.get("max_shift_m", 0.05))
        self.relearn_lost_s = float(cfg.get("relearn_lost_s", 5.0))
        self.sh_tol = tuple(float(v) for v in cfg.get("shoulder_tol_m", (0.03, 0.08)))
        self.hip_tol = tuple(float(v) for v in cfg.get("hip_tol_m", (0.02, 0.06)))
        self.rot_tol = tuple(np.deg2rad(float(v)) for v in cfg.get("rot_tol_deg", (5.0, 15.0)))
        self.follow = float(cfg.get("follow", 0.05))
        self.rigid_tol = float(cfg.get("rigid_tol_m", 0.03))
        self.adopt_s = float(cfg.get("adopt_s", 1.0))
        self.reset()

    def reset(self):
        self.R = self.origin = None
        self.points = {}       # chỉ số landmark -> vị trí tham chiếu (vai, hông), fusion triangulate
        self.weights = {}      # khung vừa rồi: trọng số điểm đo (1 = dùng đo, 0 = dùng tham chiếu)
        self._samples, self._t0, self._lost_since = [], None, None
        self._rigid_since = None
        self.hint = "dang hoc"

    @property
    def ready(self):
        return self.R is not None

    def visible(self, vis):
        """Đủ rõ để học: hai vai (và hai hông nếu require_hips)."""
        ids = (L_SH, R_SH, L_HIP, R_HIP) if self.require_hips else (L_SH, R_SH)
        return bool(min(vis[i] for i in ids) >= self.min_vis)

    def lost(self, t):
        """Khung không thấy người. Tham chiếu giữ nguyên tới relearn_lost_s rồi học lại."""
        self._samples, self._t0, self._rigid_since = [], None, None
        if self._lost_since is None:
            self._lost_since = t
        if self.ready and t - self._lost_since > self.relearn_lost_s:
            self.reset()
            self._lost_since = t
            self.hint = "mat nguoi: hoc lai"

    # -- học -------------------------------------------------------------------------------------------------------
    def learn(self, R, origin, visible, t, points=None):
        """Một mẫu học. -> True khi vừa học xong. points: {chỉ số: vị trí} cùng hệ với origin."""
        self._lost_since = None
        if not self.enabled or self.ready:
            return False
        if not visible:
            self._samples, self._t0 = [], None
            self.hint = "can thay ro vai + hong" if self.require_hips else "can thay ro 2 vai"
            return False
        if self._samples:
            R0, o0, _ = self._samples[0]
            moved = rotation_distance(R0, R) > self.max_motion or (
                o0 is not None and origin is not None and np.linalg.norm(origin - o0) > self.max_shift_m)
            if moved:
                self._samples, self._t0 = [], None
        if self._t0 is None:
            self._t0 = t
        self._samples.append((np.asarray(R, float), None if origin is None else np.asarray(origin, float),
                              {i: np.asarray(X, float) for i, X in (points or {}).items()}))
        self.hint = f"dang hoc {100 * min(1.0, (t - self._t0) / max(self.learn_s, 1e-3)):.0f}%: dung yen"
        if t - self._t0 < self.learn_s or len(self._samples) < 5:
            return False
        self.R = orthonormalize(np.mean([r for r, _, _ in self._samples], axis=0))
        origins = [o for _, o, _ in self._samples if o is not None]
        self.origin = np.mean(origins, axis=0) if len(origins) == len(self._samples) else None
        for i in self._samples[0][2]:                    # điểm chỉ lấy khi MỌI mẫu đều đo được (hông bị che: bỏ)
            pts = [p.get(i) for _, _, p in self._samples]
            if all(x is not None and np.all(np.isfinite(x)) for x in pts):
                self.points[i] = np.mean(pts, axis=0)
        self._samples, self._t0 = [], None
        self.hint = "theo do"
        return True

    # -- lọc -------------------------------------------------------------------------------------------------------
    @staticmethod
    def _weight(d, tol):
        """Lệch d -> trọng số điểm đo: 1 khi d <= tol[0], 0 khi d >= tol[1], tuyến tính ở giữa."""
        return float(np.clip((tol[1] - d) / max(tol[1] - tol[0], 1e-9), 0.0, 1.0))

    def _rigid(self, meas, t):
        """Mọi điểm tham chiếu đều đo được, đều lệch > tol[0], và khoảng cách đôi một giữ nguyên (khối cứng) liên tục
        adopt_s -> người xoay / dịch thật: nhận vị trí mới. -> True khi vừa nhận."""
        ids = list(self.points)
        moved = len(ids) >= 2 and all(
            meas.get(i) is not None and np.all(np.isfinite(meas[i])) and
            np.linalg.norm(meas[i] - self.points[i]) > (self.hip_tol if i in (L_HIP, R_HIP) else self.sh_tol)[0]
            for i in ids)
        if moved:
            moved = all(abs(np.linalg.norm(meas[a] - meas[b]) - np.linalg.norm(self.points[a] - self.points[b]))
                        < self.rigid_tol for k, a in enumerate(ids) for b in ids[k + 1:])
        if not moved:
            self._rigid_since = None
            return False
        if self._rigid_since is None:
            self._rigid_since = t
        if t - self._rigid_since < self.adopt_s:
            self.hint = f"than dich chuyen? {t - self._rigid_since:.1f}s"
            return False
        for i in ids:
            self.points[i] = np.asarray(meas[i], float).copy()
        self._rigid_since = None
        self.hint = "nhan vi tri moi"
        return True

    def gate_points(self, meas, t):
        """Fusion: meas {chỉ số: điểm đo (NaN nếu không đo được)} -> {chỉ số: điểm dùng}. Cập nhật tham chiếu."""
        self._lost_since = None
        if not (self.enabled and self.ready and self.points):
            return {}
        adopted = self._rigid(meas, t)
        out, n_est = {}, 0
        for i, ref in self.points.items():
            m = meas.get(i)
            tol = self.hip_tol if i in (L_HIP, R_HIP) else self.sh_tol
            if m is None or not np.all(np.isfinite(m)):
                w = 0.0
            else:
                d = float(np.linalg.norm(m - ref))
                w = self._weight(d, tol)
                if d < tol[0]:
                    self.points[i] = ref + self.follow * (m - ref)       # trôi chậm theo khi khớp
            self.weights[i] = w
            out[i] = self.points[i] if w <= 0 else w * np.asarray(m, float) + (1 - w) * self.points[i]
            n_est += w < 0.5
        if not adopted and n_est:
            self.hint = f"uoc luong {n_est} diem (bi che)"
        elif not adopted and self._rigid_since is None:
            self.hint = "theo do"
        return out

    def gate_rotation(self, R, t):
        """1 camera: hướng khung thân đo được -> hướng dùng (slerp giữa tham chiếu và đo theo độ lệch)."""
        self._lost_since = None
        if not (self.enabled and self.ready):
            return R
        d = rotation_distance(self.R, R)
        w = self._weight(d, self.rot_tol)
        self.weights = {"R": w}
        if d < self.rot_tol[0]:
            self.R = slerp_rotation(self.R, R, self.follow)
            self._rigid_since = None
            self.hint = "theo do"
        elif w <= 0:
            # Lệch lớn kéo dài adopt_s (x2: 1 camera không kiểm tra được khối cứng): coi là người xoay thật
            self._rigid_since = t if self._rigid_since is None else self._rigid_since
            if t - self._rigid_since >= 2 * self.adopt_s:
                self.R, self._rigid_since, self.hint = np.asarray(R, float).copy(), None, "nhan huong moi"
                return self.R
            self.hint = "uoc luong huong than (bi che)"
        return slerp_rotation(self.R, R, w)

    def status(self):
        if not self.enabled:
            return "than: theo tung khung (body_ref tat)"
        return f"than: {self.hint}" + (" (b: hoc lai)" if self.ready else "")
