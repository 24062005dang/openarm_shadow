"""Fusion nhiều camera: mỗi camera MediaPipe riêng -> triangulate -> ArmObs như 1 camera."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import cv2
import numpy as np

from openarm_shadow.camera.model import check_image_size, load_calibration
from openarm_shadow.camera.multi_source import MultiSample
from openarm_shadow.core.geometry import unit
from openarm_shadow.core.rotation import rotation_distance, slerp_rotation
from openarm_shadow.core.types import ArmObs, Frame
from openarm_shadow.fusion.orientation import HandOrientationTracker
from openarm_shadow.fusion.triangulation import (REF_FX, err_conf_factor, fuse_point, reprojection_px, triangulate_weighted)
from openarm_shadow.vision.body import BodyRef, body_frame
from openarm_shadow.vision.depth import open_finger_count, palm_frame_from_depth, sample_depth
from openarm_shadow.vision.handfusion import HandShape, OrientationFusion, PalmModel, hand_forearm_angle
from openarm_shadow.vision.landmarks import (ARM_IDX, H_INDEX_MCP, H_INDEX_TIP, H_MIDDLE_MCP, H_PINKY_MCP, H_THUMB_TIP, H_WRIST, L_EL, L_HIP, L_SH, L_WR, R_EL, R_HIP, R_SH, R_WR)
from openarm_shadow.vision.perception import _shoulders


BODY_IDS = (L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP)


PALM_IDS = (H_WRIST, H_INDEX_MCP, H_MIDDLE_MCP, 13, H_PINKY_MCP)




# ----------------------------------------------------------------------------------------------------------
# Perception nhiều camera
# ----------------------------------------------------------------------------------------------------------
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
        oc = fc.get("orientation", {})
        self.trackers = {s: HandOrientationTracker(**oc) for s in ("right", "left")}   # (cũ, giữ cho test)
        self.orient = {s: OrientationFusion(**fc.get("orientation_fusion", {})) for s in ("right", "left")}
        # body_source: "triangulate" = vai/khuỷu/cổ tay triangulate từ mọi camera (mọi camera chạy Pose);
        # "front" = lấy từ Pose của camera 0 (điểm world MediaPipe), camera khác chỉ chạy Hand -> nhẹ, mượt.
        self.body_source = str(fc.get("body_source", "triangulate"))
        hc = fc.get("hand", {})
        # Trọng số tin cậy từng camera (fusion.cameras[i].weight, mặc định 1): webcam mờ đặt thấp hơn.
        cams_cfg = fc.get("cameras") or []
        self.view_weight = [float(cams_cfg[i].get("weight", 1.0)) if i < len(cams_cfg) else 1.0
                            for i in range(len(cameras))]
        self.hand_gate_px = float(hc.get("wrist_gate_px", 60.0))
        self.palm_gate_px = float(hc.get("palm_gate_px", 18.0))
        self.hand_ref_px = float(hc.get("size_ref_px", 35.0))
        self.max_hand_forearm = float(hc.get("max_hand_forearm_deg", 100.0))
        self.wrist_from_hand = float(hc.get("wrist_from_hand", 0.7))
        self.wrist_agree_m = float(hc.get("wrist_agree_m", 0.08))
        self.shapes = {s: HandShape() for s in ("right", "left")}
        self._wrist_blend = {s: 0.0 for s in ("right", "left")}
        self.palms = {s: PalmModel(s, max_rms_m=float(hc.get("palm_max_rms_m", 0.012)))
                      for s in ("right", "left")}
        self._facing = {s: None for s in ("right", "left")}
        self._body_R, self._body_reject = None, 0
        # Tham chiếu thân (orientation.body_ref): triangulate -> mọi điểm cùng hệ camera 0, lọc từng điểm vai/hông.
        # Chế độ front lấy khung thân từ Perception camera 0 (có khoá riêng).
        self.body_ref = BodyRef(body_ref_cfg)
        self.view_frames = []
        # Khoá cùng 1 người ở mọi camera (_match_operator): vai 3D hợp nhất gần nhất + thời điểm
        self.person_cfg = dict(fc.get("person_match") or {})
        self._operator_ref = None
        self._sizes = None
        self.person_match = {}
        self.pool = ThreadPoolExecutor(len(per_view)) if parallel and len(per_view) > 1 else None

    @classmethod
    def from_config(cls, cfg, first_sample=None):
        from openarm_shadow.vision.perception import Perception
        fc = cfg["fusion"]
        names = [c["name"] for c in fc["cameras"]]
        from openarm_shadow.config import ROOT
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
        """Phím b: học lại khung thân (khoá của fusion và của từng camera)."""
        self.body_ref.reset()
        for p in self.per_view:
            if hasattr(p, "relearn_body"):
                p.relearn_body()

    def body_status(self):
        if getattr(self, "body_source", "triangulate") == "front" and self.per_view:
            return self.per_view[0].body_status() + " (camera 0)"
        return self.body_ref.status()

    def body_ready(self):
        if getattr(self, "body_source", "triangulate") == "front" and self.per_view:
            return self.per_view[0].body_ready()
        return not self.body_ref.enabled or self.body_ref.ready

    def close(self):
        for p in self.per_view:
            if hasattr(p, "close"):
                p.close()
        if self.pool is not None:
            self.pool.shutdown(wait=False)

    # -- khoá cùng 1 người ở mọi camera ------------------------------------------------------------------
    def _match_operator(self, v, sample, cands):
        """Camera phụ v: chọn trong `cands` (pose_2d từng người MediaPipe thấy ở camera v) đúng người điều khiển mà
        camera tham chiếu đang khoá, để triangulate không ghép vai/khuỷu của hai người khác nhau.
        -> chỉ số người, -1 (không ai khớp: camera v coi như không thấy ai lần này) hoặc None (chưa có tham chiếu:
        camera v tự khoá như khi chạy 1 camera).

        1. Có vai 3D hợp nhất gần đây: chiếu vào camera v, chọn người có tâm 2 vai gần điểm dự đoán nhất
           (< lock_dist x bề rộng vai shoulder_m ở khoảng cách đó). Khoá theo 3D nên người đứng sau / cạnh người điều
           khiển không lọt vào dù cùng độ cao.
        2. Chưa có (mới chạy, vừa mất người): ghép với người camera 0 đang khoá - triangulate vai/khuỷu/hông, lấy
           sai số chiếu lại nhỏ nhất. Hai camera cùng độ cao thì người khác cùng độ cao vẫn triangulate được (điểm nằm
           trên đường epipolar), nên còn đòi bề rộng vai 3D hợp lý và depth camera v đo được (nếu có) khớp.
        """
        pc = self.person_cfg
        cam = self.cams[v]
        h, w = sample.bgr.shape[:2]
        if self._operator_ref is not None:
            S = self._operator_ref[0][[L_SH, R_SH]]
            P = cam.project(S)
            z = float(cam.to_cam(S.mean(0)[None])[0, 2])
            if np.all(np.isfinite(P)) and z > 0.3:
                scale = cam.fx * float(pc.get("shoulder_m", 0.35)) / z
                d = [np.linalg.norm(sh[0] * [w, h] - P.mean(0)) / scale if sh is not None else np.inf
                     for sh in map(_shoulders, cands)]
                k = int(np.argmin(d))
                ok = d[k] < float(pc.get("lock_dist", 1.0))
                self.person_match[v] = {"mode": "3D", "ok": bool(ok), "err": float(d[k])}
                return k if ok else -1
        f0 = self.view_frames[0] if self.view_frames else None
        if f0 is None or f0.pose_2d is None or self._sizes is None:
            self.person_match[v] = {"mode": "tu khoa", "ok": True}
            return None
        cam0, P0, size0 = self.cams[0], f0.pose_2d, self._sizes[0]
        dc = self.depth_cfg
        # Bề rộng vai (m) của người camera 0 khoá, từ điểm world MediaPipe: người khác cùng độ cao triangulate ra bề
        # rộng sai (giao 2 tia nhìn ở độ sâu khác) -> bị loại kể cả khi chưa có depth.
        sr, sl = f0.arms["right"].s, f0.arms["left"].s
        ref_w = float(np.linalg.norm(sr - sl)) if sr is not None and sl is not None else None
        best = None
        for k, cand in enumerate(cands):
            use = [i for i in (L_SH, R_SH, L_EL, R_EL, L_HIP, R_HIP)
                   if P0[i, 2] >= self.min_vis and cand[i, 2] >= self.min_vis]
            if L_SH not in use or R_SH not in use or len(use) < 3:
                continue
            n0 = cam0.normalize(P0[use, :2] * size0)
            nv = cam.normalize(cand[use, :2] * [w, h])
            X, errs = {}, []
            for j, i in enumerate(use):
                Xi = triangulate_weighted([(cam0, n0[j], 1.0), (cam, nv[j], 1.0)])
                if Xi is None or not np.all(np.isfinite(Xi)):
                    errs.append(np.inf)
                    continue
                X[i] = Xi
                errs.append(max(reprojection_px(cam0, Xi, n0[j]), reprojection_px(cam, Xi, nv[j])))
            if L_SH not in X or R_SH not in X:
                continue
            width = float(np.linalg.norm(X[L_SH] - X[R_SH]))
            lo, hi = pc.get("shoulder_width_m", (0.2, 0.6))
            if not lo <= width <= hi or (ref_w and abs(width / ref_w - 1) > float(pc.get("width_tol", 0.35))):
                continue
            zc = cam.to_cam(np.array([X[L_SH], X[R_SH]]))[:, 2]
            z0 = cam0.to_cam(np.array([X[L_SH], X[R_SH]]))[:, 2]
            if min(zc.min(), z0.min()) < 0.3:
                continue
            if sample.depth_m is not None:
                dz = []
                for i, zi in zip((L_SH, R_SH), zc):
                    zm = sample_depth(sample.depth_m, cand[i, 0] * w, cand[i, 1] * h,
                                      radius=dc.get("patch_radius", 3), min_m=dc.get("min_depth_m", 0.3),
                                      max_m=dc.get("max_depth_m", 6.0), max_delta_m=dc.get("max_local_delta_m", 0.15))
                    if zm is not None:
                        dz.append(abs(zm - zi))
                if dz and min(dz) > float(pc.get("depth_tol_m", 0.25)):
                    continue
            err = float(np.median(errs))
            if best is None or err < best[0]:
                best = (err, k)
        ok = best is not None and best[0] <= float(pc.get("max_err_px", 30.0))
        self.person_match[v] = {"mode": "cam0", "ok": bool(ok), "err": best[0] if best else float("nan")}
        return best[1] if ok else -1

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
            if side not in ("right", "left") or side in o["hand"]:
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
                for s in self.orient:
                    self.orient[s].update(None)
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
            self._gate_hands(side, W[i_w], views_obs, info)
            hand_root = self._fuse_hand(side, ob, views_obs, Rb, info, W[i_e], W[i_w])
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
                # độ tin cậy riêng vai/khuỷu/cổ tay: Kalman điểm (pipeline) tin dự đoán ở điểm nhìn kém
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
            for s in self.orient:
                self.orient[s].update(None)
                arms[s].hand_orientation_mode = "NONE"
            return Frame(arms, None, [], None, t, fusion=info), W
        Rb = f0.body_R
        for side in ("right", "left"):
            src, ob = f0.arms[side], arms[side]
            ob.s, ob.e, ob.w = src.s, src.e, src.w
            ob.conf["upper"], ob.conf["fore"] = src.conf["upper"], src.conf["fore"]
            if "points" in src.conf:
                ob.conf["points"] = src.conf["points"]
            fore = Rb @ (src.w - src.e) if src.w is not None and src.e is not None else None
            self._gate_front(side, f0, views_obs, info)
            self._fuse_hand(side, ob, views_obs, Rb, info, np.zeros(3) if fore is not None else None, fore)
            if ob.grip is None and src.grip is not None:
                ob.grip = src.grip
        return Frame(arms, None, [], Rb, t, depth_used={}, body_origin=f0.body_origin, fusion=info), W

    def _gate_front(self, side, f0, views_obs, info):
        """Chế độ front: bàn tay ở camera phụ (chỉ Hand, tự gán) phải khớp cổ tay Pose của camera 0:
        - camera 0 cũng thấy bàn tay: triangulate cổ tay bàn tay từ 2 camera, sai số chiếu lại lớn -> camera phụ
          đang thấy tay khác -> bỏ;
        - camera 0 không thấy bàn tay: cổ tay từ depth của camera phụ chiếu vào camera 0 phải gần cổ tay Pose;
          không có depth thì không kiểm chứng được -> bỏ."""
        rejected = 0
        cam0 = self.cams[0]
        size0 = views_obs[0].get("size")
        wr0 = None
        if f0.pose_2d is not None and size0 is not None:
            wr0 = f0.pose_2d[ARM_IDX[side][2], :2] * size0
        h0 = views_obs[0]["hand"].get(side)
        for v in range(1, len(views_obs)):
            hv = views_obs[v]["hand"].get(side)
            if hv is None:
                continue
            ok = False
            if h0 is not None:
                # Cả 5 điểm lòng bàn tay (không chỉ cổ tay) phải khớp giữa 2 camera: một bàn tay khác nằm đúng trên
                # tia nhìn của camera 0 qua cổ tay vẫn khớp được 1 điểm, nhưng không khớp được cả lòng bàn tay.
                errs = []
                for j in PALM_IDS:
                    X = triangulate_weighted([(cam0, h0[0][j], 1.0), (self.cams[v], hv[0][j], 1.0)])
                    errs.append(np.inf if X is None else max(reprojection_px(cam0, X, h0[0][j]),
                                                             reprojection_px(self.cams[v], X, hv[0][j])))
                # Giới hạn: bàn tay khác CÙNG tư thế nằm đúng sau lưng trên tia nhìn camera 0 vẫn khớp (sai ~8 px):
                # 2 camera RGB không phân biệt được độ sâu dọc tia nhìn này; khác hướng >= ~60° thì bị loại.
                ok = float(np.median(errs)) <= self.palm_gate_px
            elif wr0 is not None and hv[2][H_WRIST] is not None:
                p0 = cam0.project(np.asarray(hv[2][H_WRIST])[None])[0]
                ok = bool(np.all(np.isfinite(p0))) and \
                    np.linalg.norm(p0 - wr0) * REF_FX / max(cam0.fx, 1e-6) <= self.hand_gate_px
            if not ok:
                del views_obs[v]["hand"][side]
                rejected += 1
        info[f"hand_{side}_rejected"] = rejected

    def _orientation_extras(self, side, views_obs, seen):
        """Nguồn phụ cho hướng bàn tay (khung world): MediaPipe world từng camera, lòng bàn tay từ depth."""
        extras = []
        for v in seen:
            cam = self.cams[v]
            f = self.view_frames[v] if v < len(self.view_frames) else None
            R_cam = getattr(f.arms[side], "hand_R_cam", None) if f is not None else None
            if R_cam is not None:
                extras.append((cam.R.T @ R_cam, 0.5, f"rgb:{cam.name}"))
            lifted = views_obs[v]["hand"][side][2]
            P = np.full((21, 3), np.nan)
            for j in PALM_IDS:
                if lifted[j] is not None:
                    P[j] = lifted[j]
            Rd, _ = palm_frame_from_depth(P, side=side)
            if Rd is not None:
                extras.append((Rd, 0.6, f"depth:{cam.name}"))
        return extras

    def _gate_hands(self, side, wrist_w, views_obs, info):
        """Bỏ bàn tay ở camera nào mà cổ tay của nó xa cổ tay Pose (đã triangulate) chiếu vào camera đó."""
        rejected = 0
        if not np.all(np.isfinite(wrist_w)):
            info[f"hand_{side}_rejected"] = 0
            return
        for v, o in enumerate(views_obs):
            if side not in o["hand"]:
                continue
            cam = self.cams[v]
            pw = cam.project(np.asarray(wrist_w)[None])[0]
            d = np.linalg.norm(o["hand"][side][3][H_WRIST] - pw) * REF_FX / max(cam.fx, 1e-6)
            if not np.isfinite(d) or d > self.hand_gate_px:
                del o["hand"][side]
                rejected += 1
        info[f"hand_{side}_rejected"] = rejected

    def _fuse_hand(self, side, ob, views_obs, Rb, info, elbow_w=None, wrist_w=None):
        """Triangulate 21 điểm, lọc đốt bất thường, hướng lòng bàn tay (Kabsch), kiểm tra với cẳng tay.
        Trả vị trí cổ tay của bàn tay (world) hoặc None."""
        seen = [v for v, o in enumerate(views_obs) if side in o["hand"]]
        if not seen:
            ob.hand_orientation_mode = self.orient[side].update(None)[1]
            info[f"hand_{side}"] = {"views": 0, "quality": 0.0, "mode": ob.hand_orientation_mode, "points": 0,
                                    "rejected": info.get(f"hand_{side}_rejected", 0)}
            return None
        facing = self._facing[side]
        pts = np.full((21, 3), np.nan)
        confs = np.zeros(21)
        stereo_palm = 0                      # số điểm lòng bàn tay thật sự được 2 camera đồng ý
        for j in range(21):
            obs, dep = [], []
            for v in seen:
                norm, conf, lifted, _, trust = views_obs[v]["hand"][side]
                tv = trust * (1.0 if facing is None else 0.25 + 0.75 * facing[v])
                obs.append((self.cams[v], norm[j], conf, tv))
                if lifted[j] is not None:
                    dep.append((self.cams[v], lifted[j], conf * tv))
            X, c, pinfo = fuse_point(obs, dep, self.reproj_px, self.depth_w, self.depth_tol)
            if X is not None:
                pts[j], confs[j] = X, c
                if j in PALM_IDS and pinfo["views"] >= 2 and not pinfo["conflict"] and \
                        not (pinfo["depth"] and pinfo["err_px"] > self.reproj_px):
                    stereo_palm += 1
        pts, dropped = self.shapes[side].filter(pts)
        confs[~np.all(np.isfinite(pts), axis=1)] = 0.0
        # Số ngón xoè: từ điểm 3D; thiếu điểm (chỉ 1 camera thấy, không depth) thì lấy từ điểm world của từng camera
        ob.hand_open_fingers = max([open_finger_count(pts)] +
                                   [self.view_frames[v].arms[side].hand_open_fingers for v in seen
                                    if v < len(self.view_frames)])
        # Kẹp có độ tin cậy riêng (4 điểm nó dùng), KHÔNG theo độ tin cậy hướng lòng bàn tay: khi chụm/xoè, lòng
        # bàn tay đổi dáng/bị che nên hướng tay hay về HOLD/ACQUIRE (tin cậy thấp) đúng lúc kẹp cần chạy.
        grip_ids = [H_WRIST, H_MIDDLE_MCP, H_THUMB_TIP, H_INDEX_TIP]
        # Tỉ số riêng từng camera (điểm world MediaPipe của camera đó): đang kẹp thì chỉ nhả khi camera thấy "chụm
        # nhất" cũng báo mở (1 camera đặt sai đầu ngón khi tay di chuyển làm khoảng cách 3D phồng ra).
        ob.grip_views = [float(self.view_frames[v].arms[side].grip) for v in sorted(seen)
                         if v < len(self.view_frames) and self.view_frames[v].arms[side].grip is not None]
        if np.all(np.isfinite(pts[grip_ids])):
            ob.grip = float(np.linalg.norm(pts[H_THUMB_TIP] - pts[H_INDEX_TIP]) /
                            max(np.linalg.norm(pts[H_MIDDLE_MCP] - pts[H_WRIST]), 1e-6))
            ob.conf["grip"] = float(np.min(confs[grip_ids]))
        else:
            # Đầu ngón bị bỏ ở 3D (thiếu camera/depth, dáng bất thường): lấy tỉ số từ điểm world MediaPipe của camera
            # thấy bàn tay, ưu tiên camera tham chiếu (tỉ số không phụ thuộc thang đo nên dùng thẳng được).
            for v in sorted(seen):
                vo = self.view_frames[v].arms[side] if v < len(self.view_frames) else None
                if vo is not None and vo.grip is not None:
                    ob.grip, ob.conf["grip"] = vo.grip, float(views_obs[v]["hand"][side][1])
                    break
        prev_q = max((self._facing[side][v] for v in seen), default=0.0) if self._facing[side] else 0.0
        raw_R, center, fit_rms, fit_mode = self.palms[side].estimate(pts, confs[list(PALM_IDS)], prev_q)
        quality, face_v = 0.0, [0.0] * len(self.cams)
        anat = hand_forearm_angle(raw_R, elbow_w, wrist_w)
        if raw_R is not None and np.isfinite(anat) and anat > self.max_hand_forearm:
            raw_R, fit_mode = None, "ANAT"          # bàn tay gập quá mức cổ tay người làm được: quan sát sai
        if raw_R is not None:
            for v, cam in enumerate(self.cams):
                face_v[v] = abs(float(raw_R[:, 2] @ unit(center - cam.center)))
            quality = max(face_v[v] for v in seen)
            self._facing[side] = face_v
        extras = [(Re, c, n) for Re, c, n in self._orientation_extras(side, views_obs, seen)
                  if not (hand_forearm_angle(Re, elbow_w, wrist_w) > self.max_hand_forearm)]
        # Độ tin cậy của nguồn 3D: trung bình các điểm lòng bàn tay (đã gồm phạt khi hai camera mâu thuẫn), giảm
        # khi chỉ còn 3 điểm; trust của camera chỉ dùng để trộn, không hạ độ tin cậy.
        pc = confs[list(PALM_IDS)]
        n_ok = int(np.count_nonzero(pc > 0))
        palm_conf = float(np.mean(pc[pc > 0]) * min(1.0, n_ok / 4)) if raw_R is not None and n_ok else 0.0
        R, mode, o_conf, used = self.orient[side].update(raw_R, max(palm_conf, 0.05), extras,
                                                         base_strong=raw_R is not None and stereo_palm >= 3)
        ob.hand_orientation_mode = mode if (raw_R is not None or mode not in ("HOLD",)) else f"HOLD {fit_mode}"
        ob.conf["hand"] = min(o_conf, palm_conf) if (used and used[0] == "3d") else o_conf
        if R is not None:
            ob.H = Rb.T @ R
            ob.hand_R_cam = R
            ob.hand_center_cam = center
            if center is not None:
                axes = np.array([center] + [center + R[:, k] * 0.08 for k in range(3)])
                px = self.cams[0].project(axes)
                if np.all(np.isfinite(px)):
                    ob.hand_axes_px = px
        info[f"hand_{side}"] = {"views": len(seen), "quality": quality, "mode": ob.hand_orientation_mode,
                                "points": int(np.count_nonzero(np.all(np.isfinite(pts), axis=1))),
                                "fit": fit_mode, "fit_mm": 1000 * fit_rms if np.isfinite(fit_rms) else np.nan,
                                "bones_dropped": dropped, "forearm_deg": anat,
                                "rejected": info.get(f"hand_{side}_rejected", 0), "sources": used}
        root = pts[H_WRIST]
        return root if np.all(np.isfinite(root)) and confs[H_WRIST] > 0 else None

    def process(self, msample: MultiSample) -> Frame:
        for cam, s in zip(self.cams, msample.views):
            if cam.rs_intr is None and s.intrinsics is not None:
                cam.set_realsense_intrinsics(s.intrinsics)
        stale = msample.stale or [False] * len(msample.views)
        if self.person_cfg.get("enabled", True):
            if self._operator_ref is not None and \
                    msample.t - self._operator_ref[1] > float(self.person_cfg.get("ref_keep_s", 0.5)):
                self._operator_ref = None            # mất người quá lâu: ghép lại với người camera 0 đang khoá
            for v in range(1, len(self.per_view)):
                self.per_view[v].operator_match = partial(self._match_operator, v, msample.views[v])
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
        self._sizes = [np.array([s.bgr.shape[1], s.bgr.shape[0]], float) for s in msample.views]
        if np.all(np.isfinite(W[[L_SH, R_SH]])):
            self._operator_ref = (W.copy(), msample.t)
        fr.fusion["person"] = {self.cams[v].name: m for v, m in self.person_match.items()}
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

        Có tham chiếu thân (body_ref): không vẽ vai/hông thô của MediaPipe (nhảy khi tay che thân) mà vẽ thân đang dùng
        sau lọc (xanh ngọc; điểm đang ước lượng vì bị che: cam, viền) và cánh tay hợp nhất vai -> khuỷu -> cổ tay (tím)
        = đúng điểm robot đang dùng."""
        from openarm_shadow.display.viz import draw_human, pixel, put_lines
        tiles = []
        W = getattr(self, "last_world", None) if world is None else world
        view_frames = self.view_frames if view_frames is None else view_frames
        ref = getattr(self, "body_ref", None)
        torso_ref = ref is not None and ref.ready and L_SH in ref.points and W is not None
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
                        q = pixel(p, img)          # điểm sát mặt phẳng camera: toạ độ khổng lồ -> bỏ, không vẽ
                        if q is not None:
                            cv2.circle(img, q, 7, (255, 0, 255), 2)
            put_lines(img, [f"{self.cams[v].name}"], org=(10, img.shape[0] - 14))
            scale = height / img.shape[0]
            tiles.append(cv2.resize(img, (int(img.shape[1] * scale), height)))
        return np.hstack(tiles)
