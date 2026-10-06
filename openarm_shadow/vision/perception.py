"""MediaPipe Pose + Hand (Tasks API, CPU/GPU) -> quan sát tay người trong khung thân."""
from __future__ import annotations

import contextlib
import threading
import time

import cv2
import numpy as np

from openarm_shadow.core.rotation import rotation_distance, slerp_rotation
from openarm_shadow.core.types import ArmObs, Frame
from openarm_shadow.vision.body import BodyRef, body_frame
from openarm_shadow.vision.depth import (deproject_pixel, fit_palm_plane, fuse_hand_landmarks, open_finger_count, palm_frame_from_depth, project_point, sample_depth)
from openarm_shadow.vision.handfusion import assign_hands_to_wrists
from openarm_shadow.vision.landmarks import (ARM_IDX, H_INDEX_MCP, H_INDEX_TIP, H_MIDDLE_MCP, H_PINKY_MCP, H_THUMB_TIP, H_WRIST, L_EL, L_HIP, L_SH, L_WR, NOSE, R_EL, R_HIP, R_SH, R_WR)


# MediaPipe GPU (TFLite GPU delegate qua EGL) bị treo vĩnh viễn khi nhiều landmarker GPU chạy đồng thời ở nhiều luồng
# (fusion 3 camera: 3 Pose + 3 Hand song song -> cả 6 lệnh detect đứng trong _process_video_data, log
# "Tensors are designed for single writes"). Mọi lệnh detect trên GPU đi qua khoá chung này (mỗi lệnh ~5 ms trên
# RTX 3050 nên chạy nối tiếp vẫn đủ nhanh); landmarker CPU vẫn chạy song song như cũ.
_GPU_LOCK = threading.Lock()


def _shoulders(pose_2d):
    """(tâm 2 vai, độ rộng vai) theo toạ độ ảnh chuẩn hoá, hoặc None nếu vai không thấy."""
    a, b = pose_2d[L_SH], pose_2d[R_SH]
    if min(a[2], b[2]) < 0.3:
        return None
    return 0.5 * (a[:2] + b[:2]), float(np.linalg.norm(a[:2] - b[:2]))


def select_operator(cands, prev, lock_dist=1.0):
    """Chọn người điều khiển trong các người Pose thấy (không nhảy sang người khác trong khung).

    - Đang khoá (prev = pose người điều khiển lần trước): chọn người có tâm 2 vai gần tâm cũ nhất, chỉ nhận nếu cách
      tâm cũ < lock_dist x độ rộng vai cũ và độ rộng vai không đổi quá 2 lần. Không ai thoả -> None (coi như hụt:
      giữ ngắn rồi mất người, robot đứng yên) thay vì nhận người khác.
    - Chưa khoá / vừa mất người: chọn người TO nhất (vai rộng nhất = đứng gần camera nhất), ưu tiên gần giữa ảnh và
      thấy đủ người (mũi, khuỷu, hông): người ngồi sau bàn / bị cắt nửa người bị ưu tiên thấp hơn.
    """
    info = [(_shoulders(c), k) for k, c in enumerate(cands)]
    info = [(s, k) for s, k in info if s is not None and s[1] > 1e-3]
    if not info:
        return None
    ps = _shoulders(prev) if prev is not None else None
    if ps is not None and ps[1] > 1e-3:
        best = min(info, key=lambda x: np.linalg.norm(x[0][0] - ps[0]))
        (c, w), k = best
        if np.linalg.norm(c - ps[0]) < lock_dist * ps[1] and 0.5 < w / ps[1] < 2.0:
            return k
        return None
    def score(x):
        (c, w), k = x
        center = 1.0 - 0.5 * min(1.0, abs(c[0] - 0.5) * 2)
        full = float(np.mean(np.clip(cands[k][[NOSE, L_EL, R_EL, L_HIP, R_HIP], 2], 0, 1)))
        return w * center * (0.3 + 0.7 * full)
    return max(info, key=score)[1]


class Perception:
    def __init__(self, pose_model, hand_model, num_hands=2, min_conf=0.5, depth_cfg=None,
                 orientation_cfg=None, pose_enabled=True, pose_interval=1, pose_hold_frames=0,
                 force_hand_side=None, parallel=True, delegate="cpu", max_people=2, lock_dist=1.0,
                 lock_keep_frames=15):
        """pose_enabled=False: chỉ chạy Hand (camera phụ trong fusion) -> nhẹ gần một nửa.
        pose_interval=N: Pose chạy 1/N khung, các khung khác dùng lại kết quả Pose gần nhất (Hand vẫn mỗi khung).
        pose_hold_frames: Pose hụt tối đa chừng này lần chạy thì vẫn giữ kết quả cũ (không mất tay vì Pose chớp).
        force_hand_side: chỉ điều khiển 1 tay -> bàn tay duy nhất thấy được luôn là tay này (không bị bỏ vì cổ tay
        Pose lệch hay nhãn handedness sai). Ý tưởng từ bản Openarm_Teleop của nhóm.
        parallel: chạy Hand và Pose song song (2 luồng) thay vì nối tiếp.
        delegate: "cpu" (XNNPACK), "gpu" (OpenGL ES qua EGL; trên Linux chạy cả GPU Intel/Mesa nếu driver hỗ trợ) hoặc
        "auto" (thử GPU, lỗi thì CPU). GPU không khởi tạo được -> tự quay về CPU và in cảnh báo.
        max_people / lock_dist: khoá người điều khiển (xem select_operator)."""
        from pathlib import Path
        self.pose_enabled = bool(pose_enabled)
        self.pose_interval, self.pose_hold = max(1, int(pose_interval)), int(pose_hold_frames)
        self.force_hand_side = force_hand_side
        self._pose_count, self._pose_misses, self._last_pose = 0, 0, None
        self.max_people, self.lock_dist = max(1, int(max_people)), float(lock_dist)
        self.lock_keep, self._lock = int(lock_keep_frames), None
        self.n_people = 0              # số người Pose thấy ở lần chạy gần nhất (hiển thị)
        models = (pose_model, hand_model) if self.pose_enabled else (hand_model,)
        missing = [str(m) for m in models if not Path(m).is_file()]
        if missing:
            raise SystemExit("Chưa có model MediaPipe: " + ", ".join(missing) +
                             "\nChạy: bash scripts/download_models.sh")
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision

        self.mp = mp
        RM = vision.RunningMode.VIDEO
        self.delegate = {}

        def create(name, landmarker, options_cls, model, **kw):
            want = str(delegate).lower()
            tries = ["gpu", "cpu"] if want in ("gpu", "auto") else ["cpu"]
            last = None
            for d in tries:
                try:
                    dlg = mpt.BaseOptions.Delegate.GPU if d == "gpu" else mpt.BaseOptions.Delegate.CPU
                    obj = landmarker.create_from_options(options_cls(
                        base_options=mpt.BaseOptions(model_asset_path=str(model), delegate=dlg), running_mode=RM, **kw))
                    self.delegate[name] = d
                    if d == "cpu" and want == "gpu":
                        print(f"Cảnh báo: {name} không chạy được trên GPU ({last}), dùng CPU.")
                    return obj
                except Exception as e:             # GPU không có / driver thiếu OpenGL ES 3.1 -> thử CPU
                    last = f"{type(e).__name__}: {e}"
            raise RuntimeError(f"Không tạo được {name}: {last}")

        self.pose = create("pose", vision.PoseLandmarker, vision.PoseLandmarkerOptions, pose_model,
                           num_poses=self.max_people, min_pose_detection_confidence=min_conf,
                           min_pose_presence_confidence=min_conf,
                           min_tracking_confidence=min_conf) if self.pose_enabled else None
        self.hands = create("hand", vision.HandLandmarker, vision.HandLandmarkerOptions, hand_model,
                            num_hands=num_hands, min_hand_detection_confidence=min_conf,
                            min_hand_presence_confidence=min_conf, min_tracking_confidence=min_conf)
        self._last_ts = -1
        # Camera có Pose: Hand chạy ở luồng riêng trong lúc Pose chạy (MediaPipe nhả GIL): ~60 -> ~43 ms/khung
        self._hand_pool = None
        if self.pose_enabled and parallel:
            from concurrent.futures import ThreadPoolExecutor
            self._hand_pool = ThreadPoolExecutor(1, thread_name_prefix="hand")
        self.depth_cfg = depth_cfg or {}
        self.orientation_cfg = orientation_cfg or {}
        self.body_ref = BodyRef(self.orientation_cfg.get("body_ref"))
        self._hand_depth_model = {"right": None, "left": None}
        self._hand_depth_misses = {"right": 0, "left": 0}
        self._hand_orientation = {"right": None, "left": None}
        self._orientation_tracking = {"right": False, "left": False}
        self._orientation_good = {"right": 0, "left": 0}
        self._orientation_bad = {"right": 0, "left": 0}
        self._body_R = None

    def _detect(self, name, img, ts):
        """detect_for_video của landmarker name ("pose"/"hand"); trên GPU thì nối tiếp qua _GPU_LOCK."""
        lm = self.pose if name == "pose" else self.hands
        with _GPU_LOCK if getattr(self, "delegate", {}).get(name) == "gpu" else contextlib.nullcontext():
            return lm.detect_for_video(img, ts)

    def relearn_body(self):
        """Phím b: học lại khung thân (vd đổi chỗ đứng / xoay người)."""
        if getattr(self, "body_ref", None) is not None:
            self.body_ref.reset()

    def body_status(self):
        ref = getattr(self, "body_ref", None)
        return ref.status() if ref is not None else ""

    def body_ready(self):
        """Dùng được để engage: đã học tham chiếu thân, hoặc body_ref tắt."""
        ref = getattr(self, "body_ref", None)
        return ref is None or not ref.enabled or ref.ready

    def close(self):
        if getattr(self, "_hand_pool", None) is not None:
            self._hand_pool.shutdown(wait=True)
        if self.pose is not None:
            self.pose.close()
        self.hands.close()

    def _hand_only(self, hres, t):
        """Camera chỉ chạy Hand: mỗi bàn tay gán cho force_hand_side (nếu có đúng 1 bàn tay) - ghép thêm ở fusion."""
        arms = {"right": ArmObs(), "left": ArmObs()}
        h2s = [np.array([[p.x, p.y] for p in hl]) for hl in (hres.hand_landmarks or [])]
        side = self.force_hand_side
        labels = [side] if (side is not None and len(h2s) == 1) else [None] * len(h2s)
        if side is not None and len(h2s) == 1:
            ob = arms[side]
            ob.conf["hand"] = 0.9
            if hres.hand_world_landmarks:
                # Hướng tay từ điểm world MediaPipe (trục cùng hướng camera này): nguồn phụ cho fusion hướng tay
                hw = np.array([[q.x, q.y, q.z] for q in hres.hand_world_landmarks[0]])
                ob.hand_R_cam, _ = palm_frame_from_depth(hw, side=side)
                ob.hand_open_fingers = open_finger_count(hw)
                hand_len = np.linalg.norm(hw[H_MIDDLE_MCP] - hw[H_WRIST])
                ob.grip = float(np.linalg.norm(hw[H_THUMB_TIP] - hw[H_INDEX_TIP]) / max(hand_len, 1e-6))
        return Frame(arms, None, list(zip(h2s, labels)), None, t)

    def _run_pose(self, img, ts):
        """Pose theo lịch pose_interval + giữ kết quả cũ khi hụt ngắn. -> (pose_2d, W) hoặc None."""
        self._pose_count += 1
        self._pose_held = False
        if self._last_pose is not None and self._pose_count % self.pose_interval != 0:
            return self._last_pose
        pres = self._detect("pose", img, ts)
        cands = [np.array([[p.x, p.y, p.visibility or 0.0] for p in P]) for P in (pres.pose_landmarks or [])]
        self.n_people = len(cands)
        # Khoá người: nhớ vai người điều khiển tới lock_keep_frames lần Pose liền không thấy họ (đi khuất ngắn, bị
        # che) để người khác trong khung không cướp quyền; quá lâu mới chọn lại người to nhất.
        lock = getattr(self, "_lock", None)
        # Fusion nhiều camera: camera phụ chọn đúng người camera tham chiếu đang khoá (MultiViewPerception.
        # _match_operator): trả chỉ số, -1 = không ai khớp (coi như hụt), None = chưa có tham chiếu -> tự khoá.
        match = getattr(self, "operator_match", None)
        k = match(cands) if match is not None and cands else None
        if k is None:
            k = select_operator(cands, lock[0] if lock else None, getattr(self, "lock_dist", 1.0))
        elif k < 0:
            k = None
        if k is not None:
            self._lock = (cands[k], 0)
            W = np.array([[p.x, p.y, p.z] for p in pres.pose_world_landmarks[k]])
            self._last_pose, self._pose_misses = (cands[k], W), 0
            return self._last_pose
        if lock:
            self._lock = (lock[0], lock[1] + 1) if lock[1] + 1 <= getattr(self, "lock_keep", 15) else None
        self._pose_misses += 1
        if self._last_pose is not None and self._pose_misses <= self.pose_hold:
            # Pose hụt: giữ vị trí cũ để vẫn gán được bàn tay, nhưng hạ độ thấy (x0.5) -> J1-J4 đứng yên thay vì
            # coi điểm cũ là mới (không che dead-man khi người đã ra khỏi khung).
            pose_2d, W = self._last_pose
            held = pose_2d.copy()
            held[:, 2] *= 0.5
            self._pose_held = True
            return held, W
        self._last_pose = None
        return None

    def _hold_orientation(self, side):
        self._orientation_tracking[side] = False
        self._orientation_good[side] = 0
        self._orientation_bad[side] += 1
        if (self._hand_orientation[side] is not None and
                self._orientation_bad[side] <= int(self.orientation_cfg.get("hold_frames", 6))):
            return self._hand_orientation[side], "HOLD"
        self._hand_orientation[side] = None
        return None, "NONE"

    def _stabilize_orientation(self, side, raw_R, source):
        if raw_R is None:
            return self._hold_orientation(side)
        prev = self._hand_orientation[side]
        if prev is not None:
            jump = np.rad2deg(rotation_distance(prev, raw_R))
            if jump > float(self.orientation_cfg.get("max_jump_deg", 70)):
                return self._hold_orientation(side)
        if not self._orientation_tracking[side]:
            self._orientation_good[side] += 1
            if self._orientation_good[side] < int(self.orientation_cfg.get("reacquire_frames", 2)):
                if prev is not None:
                    return prev, "HOLD"
                return None, "WAIT"
            self._orientation_tracking[side] = True
        self._orientation_bad[side] = 0
        self._orientation_good[side] = 0
        alpha = float(self.orientation_cfg.get("smoothing", 0.55))
        if prev is not None:
            fast_at = float(self.orientation_cfg.get("fast_angle_deg", 75))
            fast_alpha = float(self.orientation_cfg.get("fast_smoothing", 0.90))
            alpha += (fast_alpha - alpha) * min(jump / max(fast_at, 1e-3), 1.0)
        stable = raw_R if prev is None else slerp_rotation(prev, raw_R, alpha)
        self._hand_orientation[side] = stable
        return stable, source

    def process(self, bgr, t=None, depth_m=None, depth_intrinsics=None) -> Frame:
        """bgr: ảnh OpenCV gốc (không lật). t: thời điểm (s), mặc định time.monotonic()."""
        t = time.monotonic() if t is None else t
        ts = max(int(t * 1000), self._last_ts + 1)
        self._last_ts = ts
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)     # nhanh hơn bgr[:, :, ::-1] + copy (~8 -> ~1 ms ở 720p)
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        if not self.pose_enabled:
            return self._hand_only(self._detect("hand", img, ts), t)
        pool = getattr(self, "_hand_pool", None)
        if pool is not None:
            fut = pool.submit(self._detect, "hand", img, ts)
            try:
                pose = self._run_pose(img, ts)
            finally:
                hres = fut.result()          # luôn chờ Hand xong: không để 2 lần detect chồng nhau trên cùng model
        else:
            hres = self._detect("hand", img, ts)
            pose = self._run_pose(img, ts)
        h, w = bgr.shape[:2]

        arms = {"right": ArmObs(), "left": ArmObs()}
        if pose is None:
            self._body_R = None                     # người ra khỏi khung: lần sau nhận khung thân mới
            if getattr(self, "body_ref", None) is not None:
                self.body_ref.lost(t)
            for side in arms:
                self._stabilize_orientation(side, None, "NONE")
            return Frame(arms, None, [], None, t)
        pose_2d, W = pose[0].copy(), pose[1].copy()
        vis = pose_2d[:, 2]
        R_body, origin = body_frame(W, vis)

        # D455: lấy 3D metric tại landmark RGB. Chỉ dùng cho một tay khi đủ vai/khuỷu/cổ tay;
        # nếu một điểm depth mất thì giữ trọn bộ MediaPipe để không trộn hai hệ tọa độ.
        depth_points = {}
        R_depth = origin_depth = None
        if depth_m is not None and depth_intrinsics is not None:
            dc = self.depth_cfg
            ids = {L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP}
            for idx in ids:
                z = sample_depth(
                    depth_m, pose_2d[idx, 0] * w, pose_2d[idx, 1] * h,
                    radius=dc.get("patch_radius", 3), min_m=dc.get("min_depth_m", 0.3),
                    max_m=dc.get("max_depth_m", 6.0), max_delta_m=dc.get("max_local_delta_m", 0.15),
                )
                if z is not None:
                    depth_points[idx] = deproject_pixel(
                        depth_intrinsics, pose_2d[idx, 0] * w, pose_2d[idx, 1] * h, z)
            if L_SH in depth_points and R_SH in depth_points:
                D = W.copy()
                D[L_SH], D[R_SH] = depth_points[L_SH], depth_points[R_SH]
                dvis = vis.copy()
                if L_HIP in depth_points and R_HIP in depth_points:
                    D[L_HIP], D[R_HIP] = depth_points[L_HIP], depth_points[R_HIP]
                else:
                    dvis[L_HIP] = dvis[R_HIP] = 0.0
                R_depth, origin_depth = body_frame(D, dvis)

        # Ghép bàn tay với cổ tay Pose (không tin nhãn handedness): tổng khoảng cách nhỏ nhất, mỗi cổ tay 1 bàn
        # tay, bỏ bàn tay xa mọi cổ tay (tay người phía sau, nhận nhầm). Xem handfusion.assign_hands_to_wrists.
        sh_px = np.linalg.norm((pose_2d[L_SH, :2] - pose_2d[R_SH, :2]) * [w, h])
        h2s = [np.array([[p.x, p.y] for p in hl]) for hl in (hres.hand_landmarks or [])]
        wrists = {side: pose_2d[ARM_IDX[side][2], :2] * [w, h] for side in ("right", "left")}
        labels = assign_hands_to_wrists([h2[H_WRIST] * [w, h] for h2 in h2s], wrists, 0.35 * max(sh_px, 1.0))
        fs = self.force_hand_side
        if fs is not None and fs not in labels:
            # Chỉ điều khiển 1 tay: bàn tay chưa gán nào GẦN cổ tay đó hơn cổ tay kia và trong 0,6 vai thì nhận
            # (cổ tay Pose lệch không làm mất tay), nhưng không bao giờ nhận bàn tay đang ở cổ tay bên kia.
            other = "left" if fs == "right" else "right"
            best = None
            for i, h2 in enumerate(h2s):
                if labels[i] is not None:
                    continue
                r = h2[H_WRIST] * [w, h]
                d_fs, d_ot = np.linalg.norm(r - wrists[fs]), np.linalg.norm(r - wrists[other])
                if d_fs < d_ot and d_fs < 0.6 * max(sh_px, 1.0) and (best is None or d_fs < best[0]):
                    best = (d_fs, i)
            if best is not None:
                labels[best[1]] = fs
        hands_2d = list(zip(h2s, labels))
        hand_of = {side: i for i, side in enumerate(labels) if side is not None}

        body_candidate = R_depth if R_depth is not None else R_body
        if self._body_R is None or getattr(self, "_body_reject", 0) >= 10:
            self._body_R, self._body_reject = body_candidate, 0
        elif np.rad2deg(rotation_distance(self._body_R, body_candidate)) <= float(
                self.orientation_cfg.get("body_max_jump_deg", 45)):
            self._body_R = slerp_rotation(
                self._body_R, body_candidate, float(self.orientation_cfg.get("body_smoothing", 0.35)))
            self._body_reject = 0
        else:
            # Nhảy lớn: giữ khung cũ; lặp lại 10 khung liền (người quay thật / lần đầu nhận sai) thì nhận khung mới
            self._body_reject = getattr(self, "_body_reject", 0) + 1
        active_R = self._body_R
        ref = getattr(self, "body_ref", None)
        if ref is not None and ref.enabled:
            # Chỉ lọc HƯỚNG: điểm lúc dùng depth (camera) lúc dùng MediaPipe world (gốc ở hông), gốc không chung hệ.
            # Retarget chỉ dùng hướng các đoạn tay nên gốc theo từng khung không ảnh hưởng.
            if ref.ready:
                active_R = ref.gate_rotation(body_candidate, t)
            elif ref.learn(body_candidate, None, ref.visible(vis), t):
                active_R = ref.R
        active_origin = origin_depth if origin_depth is not None else origin
        depth_used, hand_depth = {}, {}
        for side, (i_s, i_e, i_w) in ARM_IDX.items():
            ob = arms[side]
            have_depth = R_depth is not None and all(i in depth_points for i in (i_s, i_e, i_w))
            if have_depth:
                td = lambda v: active_R.T @ (v - origin_depth)
                ob.s, ob.e, ob.w = td(depth_points[i_s]), td(depth_points[i_e]), td(depth_points[i_w])
                depth_used[side] = 3
            else:
                ta = lambda v: active_R.T @ (v - origin)
                ob.s, ob.e, ob.w = ta(W[i_s]), ta(W[i_e]), ta(W[i_w])
                depth_used[side] = 0
            ob.conf["upper"] = float(min(vis[i_s], vis[i_e]))
            ob.conf["fore"] = float(min(vis[i_e], vis[i_w]))
            # độ tin cậy riêng vai/khuỷu/cổ tay: Kalman điểm (pipeline) tin dự đoán ở điểm nhìn kém
            ob.conf["points"] = (float(vis[i_s]), float(vis[i_e]), float(vis[i_w]))
            if side in hand_of:
                hi = hand_of[side]
                hw = np.array([[p.x, p.y, p.z] for p in hres.hand_world_landmarks[hi]])
                hand_len = np.linalg.norm(hw[H_MIDDLE_MCP] - hw[H_WRIST])
                ob.grip = float(np.linalg.norm(hw[H_THUMB_TIP] - hw[H_INDEX_TIP]) / max(hand_len, 1e-6))
                # Không dùng điểm handedness: nó tụt khi mu/cạnh bàn tay quay về camera dù landmark vẫn tốt,
                # làm J5-J7 đứng hình. Độ tin cậy bàn tay = độ thấy cổ tay (tối thiểu 0,7 khi đã thấy bàn tay).
                v_w = vis[i_w] * (2.0 if getattr(self, "_pose_held", False) else 1.0)   # bỏ phần hạ do Pose hụt
                ob.conf["hand"] = float(max(v_w, 0.7)) if v_w >= 0.5 else float(v_w)
                if depth_m is not None and depth_intrinsics is not None:
                    hold_frames = int(self.depth_cfg.get("hand_model_hold_frames", 6))
                    previous_model = (self._hand_depth_model[side]
                                      if self._hand_depth_misses[side] < hold_frames else None)
                    pts, dinfo = fuse_hand_landmarks(hres.hand_landmarks[hi], depth_m, depth_intrinsics,
                                                     self.depth_cfg, previous_model)
                    if dinfo["direct"]:
                        self._hand_depth_misses[side] = 0
                    else:
                        self._hand_depth_misses[side] += 1
                    if dinfo["model"] is not None and dinfo["direct"]:
                        self._hand_depth_model[side] = dinfo["model"]
                    if self._hand_depth_misses[side] >= hold_frames:
                        self._hand_depth_model[side] = None
                    ob.hand_points_cam = pts
                    ob.hand_depth_valid = dinfo["direct"]
                    ob.hand_depth_confidence = dinfo["confidence"]
                    ob.hand_depth_mode = dinfo["mode"]
                    open_count = open_finger_count(pts)
                    ob.hand_open_fingers = open_count
                    raw_R, center = palm_frame_from_depth(pts, side=side)
                    plane_info = {"inliers": 0, "rms_m": float("inf")}
                    source = "LANDMARK"
                    if open_count >= int(self.orientation_cfg.get("min_open_fingers", 3)):
                        plane_n, plane_center, plane_info = fit_palm_plane(
                            depth_m, depth_intrinsics, hres.hand_landmarks[hi], self.depth_cfg)
                        max_rms = float(self.depth_cfg.get("palm_plane_max_rms_m", 0.012))
                        plane_quality = 0.0
                        if plane_n is not None:
                            support = min(1.0, plane_info["inliers"] /
                                          max(3 * int(self.depth_cfg.get("palm_min_points", 35)), 1))
                            plane_quality = support * np.exp(-plane_info["rms_m"] / max(max_rms, 1e-6))
                            source = "PLANE"
                        raw_R, center = palm_frame_from_depth(pts, plane_n, plane_quality, side=side)
                        if plane_center is not None:
                            center = plane_center
                    stable_R, orientation_mode = self._stabilize_orientation(side, raw_R, source)
                    ob.hand_orientation_mode = orientation_mode
                    if stable_R is not None:
                        ob.H = active_R.T @ stable_R
                        ob.hand_R_cam, ob.hand_center_cam = stable_R, center
                        if center is not None:
                            axis_len = float(self.depth_cfg.get("palm_axis_length_m", 0.08))
                            axis_points = [center] + [center + stable_R[:, j] * axis_len for j in range(3)]
                            axes_px = [project_point(depth_intrinsics, p) for p in axis_points]
                            if all(p is not None for p in axes_px):
                                ob.hand_axes_px = np.asarray(axes_px)
                    dinfo.update({"open_fingers": open_count, "orientation": orientation_mode,
                                  "plane_inliers": plane_info["inliers"], "plane_rms_m": plane_info["rms_m"]})
                    hand_depth[side] = {k: v for k, v in dinfo.items() if k != "model"}
                else:
                    # Webcam thường (không có depth): hướng bàn tay từ điểm "world" 3D của MediaPipe Hand.
                    # Trục của các điểm này cùng hướng trục camera, nên dùng chung palm_frame_from_depth
                    # (cùng quy ước x hướng ngón, z pháp tuyến ra khỏi lòng bàn tay) và cùng bộ ổn định.
                    ob.hand_open_fingers = open_finger_count(hw)
                    raw_R, _ = palm_frame_from_depth(hw, side=side)
                    stable_R, orientation_mode = self._stabilize_orientation(side, raw_R, "WORLD")
                    ob.hand_orientation_mode = orientation_mode
                    if stable_R is not None:
                        ob.H = active_R.T @ stable_R
                        ob.hand_R_cam = stable_R
                        # Không có intrinsics: chiếu trực giao (x, y world cùng chiều x, y ảnh),
                        # độ dài trục = khoảng cổ tay -> gốc ngón giữa trên ảnh.
                        uv = hands_2d[hi][0] * [w, h]
                        center_px = np.mean(uv[[H_WRIST, H_INDEX_MCP, H_MIDDLE_MCP, 13, H_PINKY_MCP]], axis=0)
                        axis_px = np.linalg.norm(uv[H_MIDDLE_MCP] - uv[H_WRIST])
                        ob.hand_axes_px = np.array(
                            [center_px] + [center_px + stable_R[:2, j] * axis_px for j in range(3)])
            else:
                self._stabilize_orientation(side, None, "NONE")
        return Frame(arms, pose_2d, hands_2d, active_R, t, depth_used, hand_depth, active_origin)
