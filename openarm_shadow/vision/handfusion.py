"""Làm sạch bàn tay 3D sau triangulation: ít nhiễu hơn, hướng bàn tay ổn định, đồng bộ với cánh tay.

Các khối (dùng trong multiview.MultiViewPerception; hàm gán tay cũng dùng cho Perception 1 camera):

- assign_hands_to_wrists: gán mỗi bàn tay MediaPipe cho cổ tay Pose gần nhất theo tổng khoảng cách nhỏ nhất,
  mỗi cổ tay tối đa 1 bàn tay, bỏ bàn tay xa mọi cổ tay (tay người khác phía sau, phát hiện nhầm).
  Ý tưởng: ATHENA (MIT, labels2d._reassign_hands_from_3d_wrists) gán tay theo cổ tay Pose chứ không theo nhãn
  handedness; ở đây viết lại cho thời gian thực.
- HandShape: học chiều dài 20 đốt xương của bàn tay người đang dùng (trung vị), khung sau đốt nào dài/ngắn bất
  thường thì bỏ điểm đầu đốt (đánh dấu thiếu, KHÔNG bịa điểm thay thế).
- PalmModel: hướng lòng bàn tay bằng Kabsch có trọng số giữa 5 điểm cứng của lòng bàn tay (cổ tay + 4 gốc ngón)
  và "khuôn" lòng bàn tay học từ chính người dùng. Dùng cả 5 điểm (không chỉ 3 như palm_frame_from_depth), thiếu
  1-2 gốc ngón vẫn chạy, bỏ điểm lệch nhất nếu lệch > outlier_m, khớp kém (rms > max_rms_m) thì bỏ khung.
  Lưu ý: lòng bàn tay gần phẳng nên ảnh gương trỏ<->út vẫn trùng với khuôn sau khi quay 180° quanh trục ngón;
  loại nhập nhằng dấu này vẫn do HandOrientationTracker (SIGN-FIX) xử lý.
- OrientationFusion: hướng bàn tay từ nhiều nguồn (3D, MediaPipe world từng camera, depth), 2 giả thuyết chống lật,
  nghi ngờ thì đóng băng cổ tay (ý tưởng từ bản Openarm_Teleop của nhóm).
- hand_forearm_angle: góc giữa hướng ngón tay và cẳng tay; cổ tay người gập tối đa ~80°, nên góc lớn hơn
  max_hand_forearm_deg là quan sát sai (bàn tay nhầm, lật) -> bỏ khung đó.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from openarm_shadow.core.geometry import unit

# Chỉ số MediaPipe Hand
WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP = 0, 5, 9, 13, 17
PALM = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
# 20 đốt: (gốc, đầu)
BONES = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20))


def assign_hands_to_wrists(hand_roots, wrists, gate):
    """hand_roots: [(x, y)] cổ tay của từng bàn tay MediaPipe; wrists: {side: (x, y) hoặc None}; gate: khoảng
    cách tối đa (cùng đơn vị). Trả list side|None theo thứ tự hand_roots, mỗi side dùng tối đa 1 lần.
    Ghép cặp theo khoảng cách tăng dần (với 2 tay x 2 cổ tay, kết quả trùng tối ưu tổng khoảng cách)."""
    pairs = []
    for i, r in enumerate(hand_roots):
        for side, wr in wrists.items():
            if wr is None or not np.all(np.isfinite(wr)):
                continue
            d = float(np.linalg.norm(np.asarray(r, float) - np.asarray(wr, float)))
            if d < gate:
                pairs.append((d, i, side))
    out, used = [None] * len(hand_roots), set()
    for d, i, side in sorted(pairs):
        if out[i] is None and side not in used:
            out[i], used = side, used | {side}
    return out


def hand_forearm_angle(R_hand, elbow, wrist):
    """Góc (độ) giữa hướng ngón tay (cột x của khung bàn tay) và cẳng tay (khuỷu -> cổ tay)."""
    if R_hand is None or elbow is None or wrist is None:
        return np.nan
    f = np.asarray(wrist, float) - np.asarray(elbow, float)
    if not np.all(np.isfinite(f)) or np.linalg.norm(f) < 1e-6:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(unit(f) @ unit(R_hand[:, 0]), -1.0, 1.0))))


class HandShape:
    """Chiều dài đốt xương bàn tay (học trung vị) và lọc điểm có đốt dài/ngắn bất thường."""

    def __init__(self, min_ratio=0.6, max_ratio=1.6, samples=60, min_samples=10):
        self.min_ratio, self.max_ratio, self.min_samples = min_ratio, max_ratio, min_samples
        self.hist = [deque(maxlen=samples) for _ in BONES]

    def lengths(self):
        return np.array([np.median(h) if len(h) >= self.min_samples else np.nan for h in self.hist])

    def filter(self, pts, learn=True):
        """pts (21,3) có thể NaN. Trả (pts_sạch, số điểm bị bỏ). learn: thêm khung này vào thống kê
        (chỉ các đốt đạt kiểm tra, để một khung sai không làm hỏng chiều dài chuẩn)."""
        p = np.array(pts, float)
        ref = self.lengths()
        dropped = 0
        for k, (a, b) in enumerate(BONES):
            if not (np.all(np.isfinite(p[a])) and np.all(np.isfinite(p[b]))):
                continue
            L = float(np.linalg.norm(p[b] - p[a]))
            if np.isfinite(ref[k]) and not (self.min_ratio * ref[k] <= L <= self.max_ratio * ref[k]):
                p[b] = np.nan
                dropped += 1
                continue
            if learn and 0.005 < L < 0.15:
                self.hist[k].append(L)
        return p, dropped


class PalmModel:
    """Hướng lòng bàn tay từ 5 điểm cứng bằng Kabsch có trọng số với khuôn học được.

    Khung trả về cùng quy ước palm_frame_from_depth (x hướng ngón, z pháp tuyến ra khỏi lòng bàn tay), nên thay
    thế trực tiếp được. Trước khi học xong khuôn (learn_frames khung tốt), dùng palm_frame_from_depth.
    """

    def __init__(self, side, learn_frames=15, max_rms_m=0.012, outlier_m=0.015):
        self.side, self.learn_frames = side, learn_frames
        self.max_rms, self.outlier = max_rms_m, outlier_m
        self._samples = []
        self.template = None           # (5,3) toạ độ trong khung lòng bàn tay, gốc = tâm

    def _learn(self, pts, R, center):
        local = (pts[list(PALM)] - center) @ R
        self._samples.append(local)
        if len(self._samples) >= self.learn_frames:
            T = np.median(np.array(self._samples), axis=0)
            self.template = T - T.mean(axis=0)
            self._samples = []

    def estimate(self, pts, weights=None, quality=1.0):
        """-> (R, center, rms_m, mode). mode: "KABSCH", "3PT" (chưa có khuôn) hoặc lý do bị loại."""
        from openarm_shadow.vision.depth import palm_frame_from_depth
        p = np.asarray(pts, float)
        if self.template is None:
            R, center = palm_frame_from_depth(p, side=self.side)
            if R is None:
                return None, None, np.nan, "MISSING"
            # Chỉ học từ khung thấy rõ lòng bàn tay (quality cao) để khuôn không mang dấu lật
            if quality >= 0.5:
                self._learn(p, R, center)
            return R, center, np.nan, "3PT"
        idx = [k for k, j in enumerate(PALM) if np.all(np.isfinite(p[j]))]
        if len(idx) < 3 or 0 not in idx:
            return None, None, np.nan, "MISSING"
        w = np.ones(len(PALM)) if weights is None else np.asarray(weights, float)
        while True:
            A = self.template[idx]
            B = p[[PALM[k] for k in idx]]
            ww = np.clip(w[idx], 1e-3, None)
            ca = (ww[:, None] * A).sum(0) / ww.sum()
            cb = (ww[:, None] * B).sum(0) / ww.sum()
            Hm = ((A - ca) * ww[:, None]).T @ (B - cb)
            U, _, Vt = np.linalg.svd(Hm)
            D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
            R = Vt.T @ D @ U.T                           # B ~ R A: cột của R = trục lòng bàn tay trong world
            res = np.linalg.norm((A - ca) @ R.T + cb - B, axis=1)
            worst = int(np.argmax(res))
            if res[worst] > self.outlier and len(idx) > 3 and idx[worst] != 0:
                idx.pop(worst)
                continue
            break
        rms = float(np.sqrt(np.mean(res ** 2)))
        if rms > self.max_rms:
            return None, None, rms, "BAD-FIT"
        center = cb - R @ ca
        return R, center, rms, "KABSCH"


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
        from openarm_shadow.core.rotation import slerp_rotation
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
