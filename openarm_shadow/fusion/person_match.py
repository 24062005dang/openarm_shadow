"""Khoá CÙNG một người điều khiển ở mọi camera, để triangulate không ghép vai/khuỷu của hai người khác nhau.

Camera tham chiếu (camera 0) tự khoá người như khi chạy 1 camera (perception.operator). Camera phụ chỉ nhận người
khớp với người đó (OperatorMatcher.match, gắn vào Perception của camera phụ qua thuộc tính operator_match).
"""
from __future__ import annotations

import numpy as np

from ..perception.depth import sample_depth
from ..perception.operator import shoulders
from ..perception.types import L_EL, L_HIP, L_SH, R_EL, R_HIP, R_SH
from .triangulation import reprojection_px, triangulate_weighted


class OperatorMatcher:
    """cams: CameraModel (camera 0 = khung thế giới). cfg: fusion.person_match. depth_cfg: fusion.depth.

    Trạng thái giữa các khung: ref (vai 3D hợp nhất gần nhất + thời điểm), frame0 (Frame camera 0 khung trước),
    sizes (kích thước ảnh từng camera khung trước), results (kết quả ghép từng camera phụ, để hiển thị)."""

    def __init__(self, cams, cfg=None, depth_cfg=None, min_vis=0.3):
        self.cams = cams
        self.cfg = dict(cfg or {})
        self.depth_cfg = depth_cfg or {}
        self.min_vis = float(min_vis)
        self.ref = None
        self.frame0 = None
        self.sizes = None
        self.results = {}

    @property
    def enabled(self):
        return bool(self.cfg.get("enabled", True))

    def begin(self, t):
        """Đầu mỗi khung: mất người quá ref_keep_s thì bỏ vai 3D cũ (ghép lại với người camera 0 đang khoá)."""
        if self.ref is not None and t - self.ref[1] > float(self.cfg.get("ref_keep_s", 0.5)):
            self.ref = None

    def update(self, W, t, frame0, sizes):
        """Cuối mỗi khung: lưu vai 3D vừa hợp nhất (nếu có), Frame camera 0 và kích thước ảnh."""
        self.frame0, self.sizes = frame0, sizes
        if np.all(np.isfinite(W[[L_SH, R_SH]])):
            self.ref = (W.copy(), t)

    def match(self, v, sample, cands):
        """Camera phụ v: chọn trong `cands` (pose_2d từng người MediaPipe thấy ở camera v) đúng người điều khiển mà
        camera tham chiếu đang khoá.
        -> chỉ số người, -1 (không ai khớp: camera v coi như không thấy ai lần này) hoặc None (chưa có tham chiếu:
        camera v tự khoá như khi chạy 1 camera).

        1. Có vai 3D hợp nhất gần đây: chiếu vào camera v, chọn người có tâm 2 vai gần điểm dự đoán nhất
           (< lock_dist x bề rộng vai shoulder_m ở khoảng cách đó). Khoá theo 3D nên người đứng sau / cạnh người điều
           khiển không lọt vào dù cùng độ cao.
        2. Chưa có (mới chạy, vừa mất người): ghép với người camera 0 đang khoá - triangulate vai/khuỷu/hông, lấy
           sai số chiếu lại nhỏ nhất. Hai camera cùng độ cao thì người khác cùng độ cao vẫn triangulate được (điểm nằm
           trên đường epipolar), nên còn đòi bề rộng vai 3D hợp lý và depth camera v đo được (nếu có) khớp.
        """
        pc = self.cfg
        cam = self.cams[v]
        h, w = sample.bgr.shape[:2]
        if self.ref is not None:
            S = self.ref[0][[L_SH, R_SH]]
            P = cam.project(S)
            z = float(cam.to_cam(S.mean(0)[None])[0, 2])
            if np.all(np.isfinite(P)) and z > 0.3:
                scale = cam.fx * float(pc.get("shoulder_m", 0.35)) / z
                d = [np.linalg.norm(sh[0] * [w, h] - P.mean(0)) / scale if sh is not None else np.inf
                     for sh in map(shoulders, cands)]
                k = int(np.argmin(d))
                ok = d[k] < float(pc.get("lock_dist", 1.0))
                self.results[v] = {"mode": "3D", "ok": bool(ok), "err": float(d[k])}
                return k if ok else -1
        f0 = self.frame0
        if f0 is None or f0.pose_2d is None or self.sizes is None:
            self.results[v] = {"mode": "tu khoa", "ok": True}
            return None
        cam0, P0, size0 = self.cams[0], f0.pose_2d, self.sizes[0]
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
        self.results[v] = {"mode": "cam0", "ok": bool(ok), "err": best[0] if best else float("nan")}
        return best[1] if ok else -1
