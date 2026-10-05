"""Hướng bàn tay từ nhiều nguồn: chống lật 180° quanh trục ngón, nghi ngờ thì đóng băng cổ tay."""
from __future__ import annotations

import numpy as np

from ..core.rotations import slerp_rotation


class OrientationFusion:
    """Hướng bàn tay từ nhiều nguồn, chống lật, nghi ngờ thì đóng băng cổ tay.

    Ý tưởng từ bản Openarm_Teleop của nhóm (multiview.OrientationFusion), viết lại:
    - Nguồn chính (base): lòng bàn tay từ điểm 3D triangulate (Kabsch). Không có thì lấy nguồn phụ tin cậy nhất.
    - Nguồn phụ (extras): hướng tay từ điểm "world" MediaPipe của từng camera, lòng bàn tay từ depth D435i.
    - Hai giả thuyết: base và base quay 180° quanh trục ngón (nhầm trỏ <-> út). Chấm điểm theo độ gần hướng khung
      trước và các nguồn phụ; đổi giả thuyết mà phải xoay > switch_deg thì cần switch_frames khung liên tiếp, trong
      lúc chờ trả hướng cũ với độ tin cậy 0 (J5-J7 đứng yên, không đi theo hướng chưa chắc).
    - Nguồn phụ lệch hướng đã chọn > max_disagree_deg bị bỏ; còn lại trộn SLERP theo độ tin cậy.
    - Làm mượt đầu ra thích ứng: xoay chậm thì mượt (alpha), xoay nhanh thì bám (fast_alpha) -> không trễ.
    - Độ tin cậy trả về: >= 2 nguồn đồng ý 0,9 (TRACKING); 1 nguồn 0,7 (DEGRADED, vẫn qua min_conf 0,6);
      HOLD/đang chờ đổi 0 (đóng băng cổ tay); mất quá hold_frames -> None (LOST).
    - Nhận lại sau khi mất (LOST, CONFLICT, hoặc HOLD rồi thấy lại ở hướng khác > acquire_jump_deg): trạng thái
      ACQUIRE, độ tin cậy 0 (cổ tay đứng yên) cho tới khi hướng ổn định (lệch < acquire_stable_deg giữa các khung)
      đủ acquire_frames khung có >= 2 nguồn đồng ý (1 nguồn tính nửa khung). Tránh robot xoay theo một hướng đoán
      sai ngay lúc vừa thấy lại tay (vd tay để ngang, nhìn cạnh).
    """

    FLIP = np.diag([1.0, -1.0, -1.0])

    def __init__(self, temporal_weight=1.5, switch_deg=100.0, switch_frames=3, max_disagree_deg=70.0,
                 alpha=0.65, fast_alpha=0.92, fast_angle_deg=60.0, hold_frames=12, acquire_frames=4,
                 acquire_stable_deg=20.0, acquire_jump_deg=45.0, **_):
        self.tw, self.switch, self.switch_frames = temporal_weight, switch_deg, int(switch_frames)
        self.max_dis, self.alpha, self.fast_alpha = max_disagree_deg, alpha, max(alpha, fast_alpha)
        self.fast_angle, self.hold = fast_angle_deg, int(hold_frames)
        self.R, self.bad, self.pending, self.pending_n = None, 0, None, 0
        self.conflicts = 0
        self.acq_frames, self.acq_stable, self.acq_jump = float(acquire_frames), acquire_stable_deg, acquire_jump_deg
        self.acquiring, self.acq_score, self.acq_prev = True, 0.0, None

    @staticmethod
    def _deg(a, b):
        return float(np.degrees(np.arccos(np.clip((np.trace(np.asarray(a).T @ np.asarray(b)) - 1) / 2, -1, 1))))

    def _miss(self):
        self.bad += 1
        self.pending, self.pending_n = None, 0
        if self.R is not None and self.bad <= self.hold:
            return self.R, "HOLD", 0.0, []
        self.R = None
        self.acquiring, self.acq_score, self.acq_prev = True, 0.0, None
        return None, "LOST", 0.0, []

    def update(self, base, base_conf=1.0, extras=(), base_strong=False):
        """base: 3x3 hoặc None; extras: [(R, conf, tên)]; base_strong: base đã được >= 2 camera xác nhận (tự nó
        tính là 2 nguồn). -> (R, trạng thái, độ tin cậy, nguồn đã dùng)."""
        obs = [(np.asarray(R, float), float(c), n) for R, c, n in extras if R is not None and c > 0]
        if base is None:
            if not obs:
                return self._miss()
            k = int(np.argmax([c for _, c, _ in obs]))
            (base, base_conf, base_name), obs = obs[k], obs[:k] + obs[k + 1:]
            base_strong = False
        else:
            base_name = "3d"
        base = np.asarray(base, float)
        cands = (base, base @ self.FLIP)
        # Vừa mất tay (HOLD) thì hướng cũ đã cũ: tin nó ít hơn để các nguồn hiện tại quyết định giả thuyết.
        tw = self.tw if self.bad == 0 else 0.2 * self.tw
        scores = []
        for c in cands:
            sc = tw * (self._deg(self.R, c) / 90.0) ** 2 if self.R is not None else 0.0
            sc += sum(w * (self._deg(o, c) / 90.0) ** 2 for o, w, _ in obs)
            scores.append(sc)
        best = cands[int(np.argmin(scores))]
        if self.R is not None and self._deg(self.R, best) > self.switch:
            # Hướng mới cách hướng cũ quá xa: lật do nhận diện hay xoay thật? Chờ vài khung cho chắc.
            if self.pending is not None and self._deg(self.pending, best) < 45:
                self.pending, self.pending_n = best, self.pending_n + 1     # bám theo khi đang xoay liên tục
            else:
                self.pending, self.pending_n = best, 1
            if self.pending_n < self.switch_frames:
                return self.R, "SWITCH?", 0.0, []
        self.pending, self.pending_n = None, 0
        agree = [(o, w, n) for o, w, n in obs if self._deg(best, o) <= self.max_dis]
        if obs and not agree:
            # Mọi nguồn phụ đều phản đối hướng vừa chọn (vd khoá nhầm giả thuyết lật sau khi mất tay): không điều
            # khiển cổ tay; lặp lại switch_frames khung thì bỏ hướng cũ để lần sau các nguồn chọn lại từ đầu.
            self.conflicts += 1
            if self.conflicts >= self.switch_frames:
                self.R, self.conflicts = None, 0
                self.acquiring, self.acq_score, self.acq_prev = True, 0.0, None
            return self.R, "CONFLICT", 0.0, []
        if self.bad > 0 and self.R is not None and self._deg(self.R, best) > self.acq_jump:
            self.acquiring, self.acq_score, self.acq_prev = True, 0.0, None   # thấy lại ở hướng khác hẳn
        self.conflicts, self.bad = 0, 0
        used = [(best, max(float(base_conf), 0.05), base_name)] + agree
        fused, total = used[0][0], used[0][1]
        for o, w, _ in used[1:]:
            fused = slerp_rotation(fused, o, w / max(total + w, 1e-9))
            total += w
        if self.R is not None:
            d = self._deg(self.R, fused)
            a = self.alpha + (self.fast_alpha - self.alpha) * float(np.clip(d / self.fast_angle, 0.0, 1.0))
            fused = slerp_rotation(self.R, fused, a)
        self.R = fused
        state = "TRACKING" if len(used) >= 2 or base_strong else "DEGRADED"
        if self.acquiring:
            step = 1.0 if state == "TRACKING" else 0.5
            stable = self.acq_prev is not None and self._deg(self.acq_prev, fused) < self.acq_stable
            self.acq_score = self.acq_score + step if stable else step
            self.acq_prev = fused
            if self.acq_score < self.acq_frames:
                return fused, "ACQUIRE", 0.0, [n for _, _, n in used]
            self.acquiring = False
        return fused, state, (0.9 if state == "TRACKING" else 0.7), [n for _, _, n in used]
