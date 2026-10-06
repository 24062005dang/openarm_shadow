"""Fusion nhiều camera: mỗi camera chạy MediaPipe riêng -> triangulation có trọng số -> ArmObs như 1 camera.

Đầu ra giống hệt Perception (Frame với ArmObs vai/khuỷu/cổ tay/hướng bàn tay trong khung thân), nên retarget,
bộ lọc và SafetyGate giữ nguyên. Khung thế giới = khung camera tham chiếu (camera đầu tiên trong
fusion.cameras, thường là camera trực diện): x phải, y xuống, z ra xa camera.

Các khối: triangulation.py (hợp nhất 1 điểm), person_match.py (cùng 1 người ở mọi camera), hand.py (bàn tay),
camera_model.py (mô hình camera + hiệu chuẩn). File này: quan sát từng camera, hợp nhất thân, điều phối.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import cv2
import numpy as np

from ..core.rotations import rotation_distance, slerp_rotation
from ..perception.body import BodyRef, body_frame
from ..perception.depth import sample_depth
from ..perception.types import (ARM_IDX, H_MIDDLE_MCP, H_WRIST, L_EL, L_HIP, L_SH, L_WR, R_EL, R_HIP, R_SH, R_WR,
                                ArmObs, Frame)
from .camera_model import check_image_size, load_calibration
from .hand import HandFuser
from .person_match import OperatorMatcher
from .triangulation import REF_FX, err_conf_factor, fuse_point

BODY_IDS = (L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP)
SIDES = ("right", "left")


def view_options(cfg):
    """Tuỳ chọn Perception cho từng camera trong fusion.cameras.

    - body_source "front": chỉ camera 0 chạy Pose (các camera khác chỉ Hand); "triangulate": mọi camera chạy Pose.
      fusion.cameras[i].pose ghi đè được (trừ khi triangulate cần Pose ở mọi camera).
    - models.pose_interval / pose_hold_frames: Pose chạy thưa + giữ kết quả khi hụt ngắn.
    - Chỉ điều khiển 1 tay: mỗi camera tìm 1 bàn tay và luôn coi là bàn tay người của tay robot đó.
    """
    fc, mc, mp = cfg["fusion"], cfg["models"], cfg.get("mapping", {})
    front = str(fc.get("body_source", "triangulate")) == "front"
    arms = list(mp.get("robot_arms", ["right", "left"]))
    force = None
    if len(arms) == 1:
        force = arms[0] if mp.get("mode", "direct") == "direct" else ("left" if arms[0] == "right" else "right")
    out = []
    for i, c in enumerate(fc["cameras"]):
        pose = True if not front else bool(c.get("pose", i == 0))
        # Camera có Pose tìm 2 bàn tay để gán theo cổ tay (không nhận nhầm tay kia); camera chỉ Hand tìm 1.
        out.append({"pose_enabled": pose, "pose_interval": int(mc.get("pose_interval", 1)),
                    "pose_hold_frames": int(mc.get("pose_hold_frames", 0)), "force_hand_side": force,
                    "num_hands": 1 if (force and not pose) else 2,
                    "max_people": int(mc.get("pose_max_people", 2)), "lock_dist": float(mc.get("pose_lock_dist", 1.0)),
                    "lock_keep_frames": int(mc.get("pose_lock_keep_frames", 15))})
        if not pose and force is None:
            print(f"Cảnh báo: camera '{c.get('name', i)}' chỉ chạy Hand nhưng đang điều khiển 2 tay -> không biết bàn "
                  "tay nào là tay nào, bàn tay ở camera này bị bỏ. Dùng --arms right (hoặc left).")
    if front and not out[0]["pose_enabled"]:
        raise SystemExit("fusion.body_source: front cần camera đầu tiên chạy Pose (fusion.cameras[0].pose: true)")
    return out


class MultiViewPerception:
    """Chạy một Perception (MediaPipe) cho mỗi camera rồi hợp nhất bằng triangulation.

    per_view: danh sách đối tượng có .process(bgr, t) -> Frame (Perception, hoặc bản giả khi test).
    cameras: CameraModel theo cùng thứ tự; camera 0 là khung thế giới.
    """

    def __init__(self, per_view, cameras, fusion_cfg=None, parallel=True, body_ref_cfg=None):
        fc = fusion_cfg or {}
        self.per_view, self.cams = per_view, cameras
        self.reproj_px = float(fc.get("reproj_thresh_px", 25.0))
        self.body_mono_conf = float(fc.get("body_mono_depth_conf", 0.5))
        self.body_err_px = tuple(float(v) for v in fc.get("body_err_conf_px", (10.0, 30.0)))
        self.depth_w = float(fc.get("depth_weight", 0.3))
        self.depth_tol = float(fc.get("depth_consistency_m", 0.04))
        self.min_vis = float(fc.get("min_visibility", 0.3))
        self.depth_cfg = fc.get("depth", {})
        # body_source: "triangulate" = vai/khuỷu/cổ tay triangulate từ mọi camera (mọi camera chạy Pose);
        # "front" = lấy từ Pose của camera 0 (điểm world MediaPipe), camera khác chỉ chạy Hand -> nhẹ, mượt.
        self.body_source = str(fc.get("body_source", "triangulate"))
        hc = fc.get("hand", {})
        # Trọng số tin cậy từng camera (fusion.cameras[i].weight, mặc định 1): webcam mờ đặt thấp hơn.
        cams_cfg = fc.get("cameras") or []
        self.view_weight = [float(cams_cfg[i].get("weight", 1.0)) if i < len(cams_cfg) else 1.0
                            for i in range(len(cameras))]
        self.hand_ref_px = float(hc.get("size_ref_px", 35.0))
        self.wrist_from_hand = float(hc.get("wrist_from_hand", 0.7))
        self.wrist_agree_m = float(hc.get("wrist_agree_m", 0.08))
        self._wrist_blend = {s: 0.0 for s in SIDES}
        self._body_R, self._body_reject = None, 0
        self.view_frames = []
        self.hands = HandFuser(cameras, fc)
        self.matcher = OperatorMatcher(cameras, fc.get("person_match"), self.depth_cfg, self.min_vis)
        # Tham chiếu thân (orientation.body_ref): triangulate -> mọi điểm cùng hệ camera 0, lọc từng điểm vai/hông.
        # Chế độ front lấy khung thân từ Perception camera 0 (có tham chiếu riêng).
        self.body_ref = BodyRef(body_ref_cfg)
        self.pool = ThreadPoolExecutor(len(per_view)) if parallel and len(per_view) > 1 else None

    @classmethod
    def from_config(cls, cfg, first_sample=None):
        from ..config import ROOT
        from ..perception.landmarker import Perception
        fc = cfg["fusion"]
        names = [c["name"] for c in fc["cameras"]]
        calib = Path(fc["calib_file"])
        cams = load_calibration(calib if calib.is_absolute() else ROOT / calib, names)
        if first_sample is not None:
            for cam, s in zip(cams, first_sample.views):
                if s.intrinsics is not None:
                    cam.set_realsense_intrinsics(s.intrinsics)
                check_image_size(cam, s.bgr)
        per_view = [Perception(cfg["models"]["pose"], cfg["models"]["hand"], min_conf=cfg["models"]["min_conf"],
                               delegate=cfg["models"].get("delegate", "cpu"),
                               orientation_cfg=cfg.get("orientation"), **opts)
                    for opts in view_options(cfg)]
        return cls(per_view, cams, fc, body_ref_cfg=(cfg.get("orientation") or {}).get("body_ref"))

    def relearn_body(self):
        """Phím b: học lại tham chiếu thân (của fusion và của từng camera)."""
        self.body_ref.reset()
        for p in self.per_view:
            if hasattr(p, "relearn_body"):
                p.relearn_body()

    def body_status(self):
        if self.body_source == "front" and self.per_view and hasattr(self.per_view[0], "body_status"):
            return self.per_view[0].body_status() + " (camera 0)"
        return self.body_ref.status()

    def body_ready(self):
        if self.body_source == "front" and self.per_view and hasattr(self.per_view[0], "body_ready"):
            return self.per_view[0].body_ready()
        return not self.body_ref.enabled or self.body_ref.ready

    def close(self):
        for p in self.per_view:
            if hasattr(p, "close"):
                p.close()
        if self.pool is not None:
            self.pool.shutdown(wait=False)

    # -- quan sát 2D + depth của từng camera ------------------------------------------------------------
    def _view_obs(self, v, sample, fr):
        cam = self.cams[v]
        h, w = sample.bgr.shape[:2]
        o = {"pose": {}, "hand": {}, "size": np.array([w, h], float)}
        depth = sample.depth_m
        intr = sample.intrinsics
        dc = self.depth_cfg

        def lift(u, vv, radius):
            if depth is None or intr is None:
                return None
            z = sample_depth(depth, u, vv, radius=radius, min_m=dc.get("min_depth_m", 0.3),
                             max_m=dc.get("max_depth_m", 6.0), max_delta_m=dc.get("max_local_delta_m", 0.15))
            if z is None:
                return None
            xy = cam.normalize([[u, vv]])[0]
            return cam.to_world(np.array([xy[0] * z, xy[1] * z, z]))

        cw = self.view_weight[v]
        if fr.pose_2d is not None:
            P = fr.pose_2d
            px = P[:, :2] * [w, h]
            norm = cam.normalize(px[list(BODY_IDS)])
            for k, i in enumerate(BODY_IDS):
                vis = float(P[i, 2])
                if vis >= self.min_vis:
                    o["pose"][i] = (norm[k], vis, lift(px[i, 0], px[i, 1], dc.get("patch_radius", 3)), px[i])
        for h2, side in fr.hands_2d:
            if side not in SIDES or side in o["hand"]:
                continue
            px = np.asarray(h2) * [w, h]
            conf = float(fr.arms[side].conf.get("hand", 0.0)) if side in fr.arms else 0.0
            if conf <= 0.0:
                conf = 0.5
            # Bàn tay nhỏ trên ảnh (xa, độ phân giải thấp) -> điểm kém chính xác -> trọng số thấp hơn
            size = np.linalg.norm(px[H_MIDDLE_MCP] - px[H_WRIST]) * REF_FX / max(cam.fx, 1e-6)
            trust = cw * float(np.clip(size / self.hand_ref_px, 0.4, 1.0))    # chỉ là trọng số tương đối
            norm = cam.normalize(px)
            lifted = [lift(px[j, 0], px[j, 1], dc.get("hand_patch_radius", 2)) for j in range(21)]
            o["hand"][side] = (norm, conf, lifted, px, trust)
        return o

    # -- hợp nhất ----------------------------------------------------------------------------------------
    def fuse(self, views_obs, t):
        if self.body_source == "front":
            return self._fuse_front(views_obs, t)
        info = {"views": len(self.cams), "points": {}}
        W = np.full((33, 3), np.nan)
        vis = np.zeros(33)
        for i in BODY_IDS:
            obs = [(self.cams[v], o["pose"][i][0], o["pose"][i][1], self.view_weight[v])
                   for v, o in enumerate(views_obs) if i in o["pose"]]
            dep = [(self.cams[v], o["pose"][i][2], o["pose"][i][1]) for v, o in enumerate(views_obs)
                   if i in o["pose"] and o["pose"][i][2] is not None]
            X, c, pi = fuse_point(obs, dep, self.reproj_px, self.depth_w, self.depth_tol, self.body_mono_conf)
            c *= err_conf_factor(pi.get("err_px", np.nan), *self.body_err_px)
            info["points"][i] = pi
            if X is not None:
                W[i], vis[i] = X, c
        arms = {"right": ArmObs(), "left": ArmObs()}
        ref = self.body_ref
        seen = bool(np.all(np.isfinite(W[[L_SH, R_SH]])) and min(vis[L_SH], vis[R_SH]) > 0)
        if not seen:
            ref.lost(t)                                  # tham chiếu giữ tới relearn_lost_s rồi học lại
            if not (ref.ready and L_SH in ref.points):
                self._body_R = None                      # mất người: lần sau nhận khung thân mới
                for s in SIDES:
                    self.hands.lost(s)
                return Frame(arms, None, [], None, t, fusion=info), W
            # Vai bị tay che hẳn: chạy tiếp với vai tham chiếu (thân gần như đứng yên)
        if not np.all(np.isfinite(W[[L_HIP, R_HIP]])):
            vis[L_HIP] = vis[R_HIP] = 0.0
        W_meas, vis_meas = W.copy(), vis.copy()          # điểm đo (để học tham chiếu / vẽ chẩn đoán)
        if ref.enabled and ref.ready and ref.points:
            # Lọc theo độ nhất quán: điểm đo khớp tham chiếu -> dùng đo; lệch xa (tay che) -> dùng tham chiếu
            if seen:
                used = ref.gate_points({i: W_meas[i] for i in ref.points}, t)
            else:                                        # không gọi gate_points: giữ đồng hồ mất người đang chạy
                used = dict(ref.points)
                ref.weights = {i: 0.0 for i in used}
            for i, X in used.items():
                W[i], vis[i] = X, max(vis[i], 0.9)
                info["points"].setdefault(i, {})["est"] = 1.0 - ref.weights.get(i, 0.0)
        R_body, origin = body_frame(W, vis)
        if ref.enabled and not ref.ready and seen:
            ref.learn(R_body, origin, ref.visible(vis_meas), t,
                      points={i: W_meas[i] for i in (L_SH, R_SH, L_HIP, R_HIP)})
        if self._body_R is None or self._body_reject >= 10:
            self._body_R, self._body_reject = R_body, 0
        elif np.rad2deg(rotation_distance(self._body_R, R_body)) <= 45:
            self._body_R, self._body_reject = slerp_rotation(self._body_R, R_body, 0.35), 0
        else:
            self._body_reject += 1                       # nhảy lớn: giữ khung cũ, 10 khung liền thì nhận khung mới
        Rb = self._body_R
        depth_used = {}
        for side, (i_s, i_e, i_w) in ARM_IDX.items():
            ob = arms[side]
            self.hands.gate_wrist(side, W[i_w], views_obs, info)
            hand_root = self.hands.fuse(side, ob, views_obs, self.view_frames, Rb, info, W[i_e], W[i_w])
            if np.all(np.isfinite(W[[i_s, i_e, i_w]])):
                # Đồng bộ tay-bàn tay: cổ tay của cẳng tay (J3/J4) và gốc khung bàn tay (J5-J7) dùng CHUNG một
                # điểm. Cổ tay của MediaPipe Hand chính xác hơn cổ tay Pose; chỉ dùng khi hai điểm gần nhau. Tỉ lệ
                # trộn đổi dần (0,15/khung) để bàn tay lúc có lúc mất không làm cổ tay (và J3/J4) giật.
                agree = hand_root is not None and np.linalg.norm(hand_root - W[i_w]) < self.wrist_agree_m
                goal = self.wrist_from_hand if agree else 0.0
                a = self._wrist_blend[side]
                a = self._wrist_blend[side] = a + float(np.clip(goal - a, -0.15, 0.15))
                if agree and a > 0:
                    W[i_w] = (1 - a) * W[i_w] + a * hand_root
                    info["points"][i_w]["hand_root"] = True
                ob.s, ob.e, ob.w = (Rb.T @ (W[k] - origin) for k in (i_s, i_e, i_w))
                ob.conf["upper"] = float(min(vis[i_s], vis[i_e]))
                ob.conf["fore"] = float(min(vis[i_e], vis[i_w]))
                # độ tin cậy riêng vai/khuỷu/cổ tay: Kalman điểm (pipeline, per_point_conf) tin dự đoán ở điểm kém
                ob.conf["points"] = (float(vis[i_s]), float(vis[i_e]), float(vis[i_w]))
            depth_used[side] = sum(info["points"][k]["depth"] > 0 for k in (i_s, i_e, i_w))
        return Frame(arms, None, [], Rb, t, depth_used=depth_used, body_origin=origin, fusion=info), W

    def _fuse_front(self, views_obs, t):
        """Vai/khuỷu/cổ tay + khung thân từ Pose camera 0 (điểm world MediaPipe, trục cùng hướng camera 0);
        bàn tay hợp nhất từ mọi camera. Giống cách bản Openarm_Teleop chạy mượt trên robot thật."""
        info = {"views": len(self.cams), "points": {}, "body": "front"}
        W = np.full((33, 3), np.nan)
        f0 = self.view_frames[0] if self.view_frames else None
        arms = {"right": ArmObs(), "left": ArmObs()}
        if f0 is None or f0.body_R is None:
            for s in SIDES:
                self.hands.lost(s)
                arms[s].hand_orientation_mode = "NONE"
            return Frame(arms, None, [], None, t, fusion=info), W
        Rb = f0.body_R
        for side in SIDES:
            src, ob = f0.arms[side], arms[side]
            ob.s, ob.e, ob.w = src.s, src.e, src.w
            ob.conf["upper"], ob.conf["fore"] = src.conf["upper"], src.conf["fore"]
            if "points" in src.conf:
                ob.conf["points"] = src.conf["points"]
            fore = Rb @ (src.w - src.e) if src.w is not None and src.e is not None else None
            self.hands.gate_front(side, f0, views_obs, info)
            self.hands.fuse(side, ob, views_obs, self.view_frames, Rb, info,
                            np.zeros(3) if fore is not None else None, fore)
            if ob.grip is None and src.grip is not None:
                ob.grip = src.grip
        return Frame(arms, None, [], Rb, t, depth_used={}, body_origin=f0.body_origin, fusion=info), W

    def process(self, msample) -> Frame:
        """msample: MultiSample (cameras.multicam / cameras.raw_replay)."""
        for cam, s in zip(self.cams, msample.views):
            if cam.rs_intr is None and s.intrinsics is not None:
                cam.set_realsense_intrinsics(s.intrinsics)
        stale = msample.stale or [False] * len(msample.views)
        if self.matcher.enabled:
            self.matcher.begin(msample.t)
            for v in range(1, len(self.per_view)):
                self.per_view[v].operator_match = partial(self.matcher.match, v, msample.views[v])
        jobs = [(p, s.bgr) for p, s, st in zip(self.per_view, msample.views, stale) if not st]
        if self.pool is not None:
            done = list(self.pool.map(lambda a: a[0].process(a[1], msample.t), jobs))
        else:
            done = [p.process(b, msample.t) for p, b in jobs]
        it = iter(done)
        # Camera stale: không chạy MediaPipe, coi như không thấy gì (không trộn khung cũ vào triangulation)
        frames = [Frame({"right": ArmObs(), "left": ArmObs()}, None, [], None, msample.t) if st else next(it)
                  for st in stale]
        self.view_frames = frames
        views_obs = [self._view_obs(v, s, f) for v, (s, f) in enumerate(zip(msample.views, frames))]
        fr, W = self.fuse(views_obs, msample.t)
        self.matcher.update(W, msample.t, frames[0],
                            [np.array([s.bgr.shape[1], s.bgr.shape[0]], float) for s in msample.views])
        fr.fusion["person"] = {self.cams[v].name: m for v, m in self.matcher.results.items()}
        fr.pose_2d, fr.hands_2d = frames[0].pose_2d, frames[0].hands_2d
        fr.fusion["skew_ms"] = 1000.0 * msample.skew_s
        fr.fusion["people"] = [p.n_people for p in self.per_view
                               if getattr(p, "pose", None) is not None and hasattr(p, "n_people")]
        fr.fusion["stale"] = [self.cams[v].name for v, st in enumerate(stale) if st]
        self.last_world = W
        return fr

    def draw(self, msample, fused: Frame, height=360, view_frames=None, world=None):
        """Ảnh từng camera (khung xương MediaPipe) + điểm hợp nhất chiếu lại (tím) để thấy hai camera có khớp.
        view_frames/world: kết quả của đúng khung msample (khi process() đang chạy khung sau ở luồng khác).

        Có tham chiếu thân (orientation.body_ref): không vẽ vai/hông thô của MediaPipe (nhảy khi tay che thân) mà vẽ
        thân đang dùng sau lọc (xanh ngọc; điểm đang ước lượng vì bị che: vòng cam) và cánh tay hợp nhất vai -> khuỷu
        -> cổ tay (tím) = đúng điểm robot đang dùng."""
        from ..viz import draw_human, put_lines
        from ..viz.draw import pixel
        tiles = []
        W = getattr(self, "last_world", None) if world is None else world
        view_frames = self.view_frames if view_frames is None else view_frames
        ref = self.body_ref
        torso_ref = ref.enabled and ref.ready and L_SH in ref.points and W is not None
        skip = (L_SH, R_SH, L_HIP, R_HIP) if torso_ref else ()
        est = {i for i, wgt in (ref.weights.items() if torso_ref else ()) if wgt < 0.5}
        for v, (s, f) in enumerate(zip(msample.views, view_frames)):
            img = draw_human(s.bgr.copy(), f if v else fused, skip=skip)
            th = max(2, img.shape[1] // 250)
            if torso_ref:
                px = {i: pixel(p, img) for i, p in zip(BODY_IDS, self.cams[v].project(W[list(BODY_IDS)]))}

                def seg(a, b, color, t):
                    if px.get(a) is not None and px.get(b) is not None:
                        cv2.line(img, px[a], px[b], color, t)
                for a, b in ((L_SH, R_SH), (L_SH, L_HIP), (R_SH, R_HIP), (L_HIP, R_HIP)):
                    seg(a, b, (255, 255, 0), th + 1)                     # thân đang dùng
                for a, b in ((L_SH, L_EL), (L_EL, L_WR), (R_SH, R_EL), (R_EL, R_WR)):
                    seg(a, b, (255, 0, 255), th)                         # cánh tay hợp nhất
                for i in (L_SH, R_SH, L_HIP, R_HIP):
                    if px.get(i) is not None and i in ref.points:
                        if i in est:                                     # bị che: đang dùng tham chiếu
                            cv2.circle(img, px[i], th + 5, (0, 140, 255), 2)
                        else:
                            cv2.circle(img, px[i], th + 4, (255, 255, 0), -1)
                put_lines(img, ["THAN" + (f" (uoc luong {len(est)} diem)" if est else "")], org=(10, 24),
                          color=(0, 140, 255) if est else (255, 255, 0))
            if W is not None:
                ids = [i for i in (L_SH, R_SH, L_EL, R_EL, L_WR, R_WR) if np.all(np.isfinite(W[i]))]
                if ids:
                    for p in self.cams[v].project(W[ids]):
                        q = pixel(p, img)
                        if q is not None:
                            cv2.circle(img, q, 7, (255, 0, 255), 2)
            put_lines(img, [f"{self.cams[v].name}"], org=(10, img.shape[0] - 14))
            scale = height / img.shape[0]
            tiles.append(cv2.resize(img, (int(img.shape[1] * scale), height)))
        return np.hstack(tiles)
