"""Hợp nhất bàn tay từ nhiều camera: gác cổng theo cổ tay, triangulate 21 điểm, lọc đốt xương, hướng lòng bàn tay
(Kabsch), kiểm tra giải phẫu, hướng tay nhiều nguồn (OrientationFusion), độ mở kẹp.

Ý tưởng (viết lại, không chép code):
- ATHENA (MIT, Mulla & Michaels 2025): bàn tay ở một camera phải nằm gần cổ tay Pose đã triangulate chiếu vào
  camera đó, không thì bỏ bàn tay ở camera đó (tay người khác, gán nhầm tay).
- Fortini et al. 2023 (trọng số theo góc nhìn) và AnyTeleop 2023 (ưu tiên camera tin cậy nhất): trọng số của
  một camera cho các điểm bàn tay tăng khi camera đó nhìn thẳng vào lòng/lưng bàn tay.
"""
from __future__ import annotations

import numpy as np

from ..core.geometry import unit
from ..perception.hand_geometry import open_finger_count, palm_frame_from_depth
from ..perception.types import ARM_IDX, H_INDEX_MCP, H_INDEX_TIP, H_MIDDLE_MCP, H_PINKY_MCP, H_THUMB_TIP, H_WRIST
from .hand_model import HandShape, PalmModel, hand_forearm_angle
from .orientation import OrientationFusion
from .triangulation import REF_FX, fuse_point, reprojection_px, triangulate_weighted

PALM_IDS = (H_WRIST, H_INDEX_MCP, H_MIDDLE_MCP, 13, H_PINKY_MCP)
SIDES = ("right", "left")


class HandFuser:
    """Trạng thái bàn tay theo từng tay người (khuôn lòng bàn tay, độ dài đốt, hướng tay, camera nào đang nhìn thẳng
    lòng bàn tay) và các bước hợp nhất. cams: CameraModel; fc: cấu hình fusion."""

    def __init__(self, cams, fc=None):
        fc = fc or {}
        hc = fc.get("hand", {})
        self.cams = cams
        self.reproj_px = float(fc.get("reproj_thresh_px", 25.0))
        self.depth_w = float(fc.get("depth_weight", 0.3))
        self.depth_tol = float(fc.get("depth_consistency_m", 0.04))
        self.hand_gate_px = float(hc.get("wrist_gate_px", 60.0))
        self.palm_gate_px = float(hc.get("palm_gate_px", 18.0))
        self.max_hand_forearm = float(hc.get("max_hand_forearm_deg", 100.0))
        self.shapes = {s: HandShape() for s in SIDES}
        self.palms = {s: PalmModel(s, max_rms_m=float(hc.get("palm_max_rms_m", 0.012))) for s in SIDES}
        self.orient = {s: OrientationFusion(**fc.get("orientation_fusion", {})) for s in SIDES}
        self._facing = {s: None for s in SIDES}

    def lost(self, side):
        """Không thấy người: hướng tay chuyển HOLD/LOST. -> trạng thái hướng tay."""
        return self.orient[side].update(None)[1]

    # -- gác cổng: bàn tay ở camera nào là của người điều khiển ------------------------------------------
    def gate_wrist(self, side, wrist_w, views_obs, info):
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

    def gate_front(self, side, f0, views_obs, info):
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

    # -- hợp nhất ----------------------------------------------------------------------------------------
    def orientation_extras(self, side, views_obs, seen, view_frames):
        """Nguồn phụ cho hướng bàn tay (khung world): MediaPipe world từng camera, lòng bàn tay từ depth."""
        extras = []
        for v in seen:
            cam = self.cams[v]
            f = view_frames[v] if v < len(view_frames) else None
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

    def fuse(self, side, ob, views_obs, view_frames, Rb, info, elbow_w=None, wrist_w=None):
        """Triangulate 21 điểm, lọc đốt bất thường, hướng lòng bàn tay (Kabsch), kiểm tra với cẳng tay; ghi vào ob
        (ArmObs): H, grip, conf hand/grip, hand_*. Trả vị trí cổ tay của bàn tay (world) hoặc None."""
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
                                   [view_frames[v].arms[side].hand_open_fingers for v in seen
                                    if v < len(view_frames)])
        # Kẹp có độ tin cậy riêng (4 điểm nó dùng), KHÔNG theo độ tin cậy hướng lòng bàn tay: khi chụm/xoè, lòng
        # bàn tay đổi dáng/bị che nên hướng tay hay về HOLD/ACQUIRE (tin cậy thấp) đúng lúc kẹp cần chạy.
        grip_ids = [H_WRIST, H_MIDDLE_MCP, H_THUMB_TIP, H_INDEX_TIP]
        if np.all(np.isfinite(pts[grip_ids])):
            ob.grip = float(np.linalg.norm(pts[H_THUMB_TIP] - pts[H_INDEX_TIP]) /
                            max(np.linalg.norm(pts[H_MIDDLE_MCP] - pts[H_WRIST]), 1e-6))
            ob.conf["grip"] = float(np.min(confs[grip_ids]))
        else:
            # Đầu ngón bị bỏ ở 3D (thiếu camera/depth, dáng bất thường): lấy tỉ số từ điểm world MediaPipe của camera
            # thấy bàn tay, ưu tiên camera tham chiếu (tỉ số không phụ thuộc thang đo nên dùng thẳng được).
            for v in sorted(seen):
                vo = view_frames[v].arms[side] if v < len(view_frames) else None
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
        extras = [(Re, c, n) for Re, c, n in self.orientation_extras(side, views_obs, seen, view_frames)
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
