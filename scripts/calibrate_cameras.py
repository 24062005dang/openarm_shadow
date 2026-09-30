#!/usr/bin/env python3
"""Hiệu chuẩn ngoại tham số các camera trong fusion.cameras bằng bảng ChArUco in ra giấy.

    python scripts/make_charuco_board.py -o charuco_a4.png     # in 100%, dán lên tấm phẳng, đo lại cạnh ô
    python scripts/calibrate_cameras.py --config config/fusion_2cam.yaml

Cầm bảng trong vùng tay sẽ cử động, sao cho MỌI camera cùng thấy. Đổi vị trí và góc nghiêng giữa các lần chụp
(chương trình tự chụp khi bảng đứng yên ở chỗ mới). Đủ số lần chụp thì tự tính và ghi fusion.calib_file.
Phím: c = tính ngay (khi đã >= 8 lần), q/Esc = thoát không lưu.
Camera không phải RealSense được hiệu chuẩn luôn nội tham số từ chính các ảnh bảng.
"""
import argparse
import datetime
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openarm_shadow.calibration import (average_extrinsics, board_pose, calibrate_intrinsics, detect, make_board,
                                        relative_extrinsic, reprojection_rms_px)
from openarm_shadow.config import ROOT, load_config
from openarm_shadow.multiview import CameraModel, MultiCameraSource
from openarm_shadow.viz import put_lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", default=None)
    ap.add_argument("--samples", type=int, default=20, help="số lần chụp (mọi camera cùng thấy bảng)")
    ap.add_argument("--min-corners", type=int, default=10)
    args = ap.parse_args()
    cfg = load_config(args.config)
    fc = cfg["fusion"]
    if len(fc["cameras"]) < 2:
        raise SystemExit("fusion.cameras cần ít nhất 2 camera (xem config/fusion_2cam.yaml)")
    names = [c["name"] for c in fc["cameras"]]
    board, det = make_board(fc["board"])
    src = MultiCameraSource(cfg)
    cams = [CameraModel(n) for n in names]
    samples, intr_dets, sizes = [], [[] for _ in cams], [None] * len(cams)
    last_t, last_ref = 0.0, None
    last_intr = [(0.0, None)] * len(cams)          # webcam: thêm ảnh riêng lẻ cho nội tham số
    cv2.namedWindow("calibrate", cv2.WINDOW_NORMAL)
    try:
        while True:
            ok, ms = src.read()
            if not ok:
                raise SystemExit("Mất khung camera")
            for v, (cam, s) in enumerate(zip(cams, ms.views)):
                sizes[v] = (s.bgr.shape[1], s.bgr.shape[0])
                if cam.rs_intr is None and s.intrinsics is not None:
                    cam.set_realsense_intrinsics(s.intrinsics)
            dets = [detect(det, s.bgr) for s in ms.views]
            counts = [0 if ids is None else len(ids) for ids, _ in dets]
            now = time.monotonic()
            all_ok = min(counts) >= args.min_corners and ms.skew_s <= src.tol
            ref_center = None if dets[0][1] is None else dets[0][1].mean(axis=0)
            moved = last_ref is None or (ref_center is not None and np.linalg.norm(ref_center - last_ref) > 40)
            if all_ok and moved and now - last_t > 0.8:
                samples.append(dets)
                last_t, last_ref = now, ref_center
            for v, (ids, px) in enumerate(dets):
                if cams[v].rs_intr is not None or ids is None or len(ids) < 12:
                    continue
                t_i, c_i = last_intr[v]
                c = px.mean(axis=0)
                if now - t_i > 0.5 and (c_i is None or np.linalg.norm(c - c_i) > 40):
                    intr_dets[v].append((ids, px))
                    last_intr[v] = (now, c)
            tiles = []
            for v, (s, (ids, px)) in enumerate(zip(ms.views, dets)):
                img = s.bgr.copy()
                if ids is not None:
                    cv2.aruco.drawDetectedCornersCharuco(img, px.reshape(-1, 1, 2).astype(np.float32),
                                                         ids.reshape(-1, 1))
                put_lines(img, [f"{names[v]}: {counts[v]} goc"], org=(10, img.shape[0] - 14))
                tiles.append(cv2.resize(img, (int(img.shape[1] * 360 / img.shape[0]), 360)))
            view = np.hstack(tiles)
            put_lines(view, [f"da chup {len(samples)}/{args.samples} | lech khung {1000 * ms.skew_s:.0f} ms",
                             "cam bang de MOI camera cung thay, doi vi tri/goc nghieng | c: tinh ngay | q: thoat"])
            cv2.imshow("calibrate", view)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                raise SystemExit("Huỷ, không lưu.")
            if len(samples) >= args.samples or (k == ord("c") and len(samples) >= 8):
                break
    finally:
        src.close()
        cv2.destroyAllWindows()

    out = {"reference": names[0], "board": fc["board"],
           "created": datetime.datetime.now().isoformat(timespec="seconds"), "cameras": {}}
    for v, cam in enumerate(cams):
        if cam.rs_intr is None:
            K, dist, rms = calibrate_intrinsics(board, intr_dets[v] + [s[v] for s in samples], sizes[v])
            cam.K, cam.dist = K, dist
            print(f"{names[v]}: nội tham số webcam, sai số {rms:.2f} px")
    out["cameras"][names[0]] = {"R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0]}
    if cams[0].rs_intr is None:
        out["cameras"][names[0]].update(K=cams[0].K.tolist(), dist=cams[0].dist.tolist())
    for v in range(1, len(cams)):
        ests = []
        for s in samples:
            p0 = board_pose(board, cams[0], *s[0])
            pv = board_pose(board, cams[v], *s[v])
            if p0 is not None and pv is not None:
                ests.append(relative_extrinsic(p0, pv))
        if len(ests) < 5:
            raise SystemExit(f"{names[v]}: chỉ {len(ests)} lần chụp dùng được, cần >= 5. Chụp lại.")
        R, t, keep = average_extrinsics(ests)
        cams[v].R, cams[v].t = R, t
        rms = reprojection_rms_px(board, cams[0], cams[v], [(s[0], s[v]) for s in samples])
        ang = np.degrees(np.arccos(np.clip(cams[0].R[2] @ R[2], -1, 1)))
        base = np.linalg.norm(cams[v].center - cams[0].center)
        entry = {"R": R.tolist(), "t": t.tolist(), "rms_px": round(rms, 2), "samples_used": int(keep.sum())}
        if cams[v].rs_intr is None:
            entry.update(K=cams[v].K.tolist(), dist=cams[v].dist.tolist())
        out["cameras"][names[v]] = entry
        verdict = "tốt" if rms < 3 else ("tạm được" if rms < 6 else "KÉM, nên chụp lại")
        print(f"{names[v]}: lệch trục nhìn {ang:.1f}°, cách camera {names[0]} {base * 100:.0f} cm, "
              f"sai số chiếu lại {rms:.2f} px ({verdict}), dùng {int(keep.sum())}/{len(ests)} lần chụp")
    path = Path(fc["calib_file"])
    path = path if path.is_absolute() else ROOT / path
    path.write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True))
    print("Đã ghi", path)


if __name__ == "__main__":
    main()
