"""Triangulate có trọng số nhiều camera + depth phân xử (hợp nhất một điểm)."""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------------------------------------
# Triangulation có trọng số
# ----------------------------------------------------------------------------------------------------------
def triangulate_weighted(obs, depth=()):
    """Bình phương tối thiểu có trọng số cho 1 điểm.

    obs: [(CameraModel, (x, y) chuẩn hoá, w)], depth: [(X_world, w)]. Mỗi camera cho 2 phương trình tuyến tính
    (dạng DLT không đồng nhất); mỗi điểm depth cho 3 phương trình X = X_depth. Trả None nếu thiếu ràng buộc.
    """
    rows, rhs = [], []
    for cam, (x, y), w in obs:
        R, t = cam.R, cam.t
        rows += [w * (x * R[2] - R[0]), w * (y * R[2] - R[1])]
        rhs += [w * (t[0] - x * t[2]), w * (t[1] - y * t[2])]
    for Xd, w in depth:
        rows += list(w * np.eye(3))
        rhs += list(w * np.asarray(Xd, float))
    if len(rows) < 3:
        return None
    A, b = np.asarray(rows), np.asarray(rhs)
    if np.linalg.matrix_rank(A, tol=1e-9) < 3:
        return None
    return np.linalg.lstsq(A, b, rcond=None)[0]


REF_FX = 600.0    # tiêu cự của ảnh 640x480 thường gặp: sai số được quy về thang này


def reprojection_px(cam, X, xy):
    """Sai số chiếu lại, tính bằng px QUY ĐỔI về ảnh 640x480 (tiêu cự 600 px) để ngưỡng reproj_thresh_px không
    phụ thuộc độ phân giải: webcam 1280x720 và D435i 640x480 dùng chung một ngưỡng (1 px ~ 0,1°)."""
    Xc = cam.to_cam(X)
    if Xc[2] <= 1e-6:
        return np.inf
    return float(np.hypot(Xc[0] / Xc[2] - xy[0], Xc[1] / Xc[2] - xy[1]) * REF_FX)


def err_conf_factor(err_px, a=10.0, b=30.0):
    """Hệ số độ tin cậy theo sai số chiếu lại (px quy đổi): 1 khi <= a, giảm tuyến tính, tối thiểu 0.3 từ b."""
    if not np.isfinite(err_px):
        return 1.0
    return float(np.clip(1.0 - (err_px - a) / max(b - a, 1e-6), 0.3, 1.0))


def fuse_point(obs, depth, reproj_px=25.0, depth_weight=0.3, depth_tol_m=0.04, mono_depth_conf=0.8):
    """Hợp nhất 1 điểm từ nhiều camera. Trả (X, conf, info).

    1. >= 2 camera: triangulate 2D; nếu sai số chiếu lại vượt ngưỡng thì bỏ camera tệ nhất (còn >= 2 camera),
       hoặc (còn 2 camera) thử "1 camera + depth của chính nó" và giữ phương án chiếu lại tốt nhất.
    2. Depth chỉ được thêm khi khớp với nghiệm 2D trong depth_tol_m. Depth lệch:
       - GẦN camera hơn nghiệm 2D = vật che phía trước (ngón che ngón) -> bỏ depth, giữ nguyên độ tin cậy;
       - XA hơn = mâu thuẫn (lỗi dọc đường epipolar mà 2 camera không tự thấy, hoặc depth rơi vào nền)
         -> info["conflict"] = True, độ tin cậy x0.5.
    3. 1 camera: chỉ dùng được nếu camera đó có depth; độ tin cậy x mono_depth_conf.

    obs: [(cam, xy, conf)] hoặc [(cam, xy, conf, trust)]. conf (độ tin cậy MediaPipe) quyết định độ tin cậy trả về;
    trust (trọng số camera, kích thước bàn tay, góc nhìn) chỉ dùng để trộn các camera khi triangulate.
    """
    obs = [o for o in obs if o[2] > 0.05 and np.all(np.isfinite(o[1]))]
    conf_of = {id(o[0]): float(o[2]) for o in obs}
    obs = [(o[0], o[1], float(o[2]) * (float(o[3]) if len(o) > 3 else 1.0)) for o in obs]
    dep = {id(o[0]): d for o, d in zip(obs, [None] * len(obs))}
    for cam, Xd, w in depth:
        if np.all(np.isfinite(Xd)) and w > 0.05:
            dep[id(cam)] = (np.asarray(Xd, float), w)
    info = {"views": len(obs), "err_px": np.nan, "depth": 0, "conflict": False}
    if len(obs) >= 2:
        use = list(obs)
        X = triangulate_weighted(use)
        errs = [reprojection_px(c, X, xy) for c, xy, _ in use] if X is not None else []
        while X is not None and max(errs) > reproj_px and len(use) > 2:
            use.pop(int(np.argmax(errs)))
            X = triangulate_weighted(use)
            errs = [reprojection_px(c, X, xy) for c, xy, _ in use] if X is not None else []
        conf = float(np.mean([conf_of[id(c)] for c, _, _ in use]))
        if X is not None and max(errs) > reproj_px:
            # 2 camera mâu thuẫn (một camera đoán sai điểm, vd bị che): depth là bằng chứng độc lập để phân xử.
            # Ứng viên = 1 camera + depth của chính nó; chọn ứng viên khớp nhất với các camera còn lại.
            cands = []
            for c, xy, w in use:
                d = dep.get(id(c))
                if d is None:
                    continue
                Xc = triangulate_weighted([(c, xy, w)], [(d[0], depth_weight * d[1])])
                if Xc is not None:
                    e_other = max([reprojection_px(c2, Xc, xy2) for c2, xy2, _ in use if c2 is not c] or [0.0])
                    cands.append((e_other, len(cands), Xc))
            if cands:
                X = min(cands)[2]
                info["depth"] = 1
                conf *= 0.5
            else:
                conf *= 0.3
        if X is None:
            return None, 0.0, info
        extra = []
        for c, _, _ in use:
            d = dep.get(id(c))
            if d is None:
                continue
            if np.linalg.norm(d[0] - X) < depth_tol_m:
                extra.append((d[0], depth_weight * d[1]))
            elif info["depth"] == 0 and c.to_cam(d[0])[2] > c.to_cam(X)[2]:
                # Depth gần hơn nghiệm 2D = mặt phía trước che điểm (ngón che ngón): bỏ depth, không sao.
                # Depth XA hơn: không thể là vật che -> một camera đoán sai điểm dọc đường epipolar (2 camera
                # không tự phát hiện được) hoặc depth rơi vào nền ở mép tay. Không đoán bên nào: hạ độ tin cậy.
                info["conflict"] = True
                conf *= 0.5
        if extra and info["depth"] == 0:
            X2 = triangulate_weighted(use, extra)
            if X2 is not None:
                X, info["depth"] = X2, len(extra)
        info["err_px"] = float(max(reprojection_px(c, X, xy) for c, xy, _ in use))
        info["views"] = len(use)
        return X, conf, info
    if len(obs) == 1 and dep.get(id(obs[0][0])) is not None:
        d = dep[id(obs[0][0])]
        info["depth"] = 1
        return d[0], mono_depth_conf * min(conf_of[id(obs[0][0])], d[1]), info
    return None, 0.0, info
