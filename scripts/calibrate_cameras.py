#!/usr/bin/env python3
"""Hiệu chuẩn ngoại tham số các camera trong fusion.cameras bằng bảng ChArUco in ra giấy.

    python scripts/make_charuco_board.py -o charuco_a4.png     # in 100%, dán lên tấm phẳng, đo lại cạnh ô
    python scripts/calibrate_cameras.py --config config/fusion_2cam.yaml

Cầm bảng trong vùng tay sẽ cử động, sao cho camera tham chiếu (camera đầu) và ít nhất 1 camera khác cùng thấy (3
camera: hai camera hai bên không cần thấy cùng lúc). Đổi vị trí và góc nghiêng giữa các lần chụp (chương trình tự
chụp khi bảng đứng yên ở chỗ mới). Mỗi camera phụ đủ số lần chụp chung với camera tham chiếu thì tự tính và ghi
fusion.calib_file.
Phím: c = tính ngay (khi đã >= 8 lần), q/Esc = thoát không lưu.
Webcam (không phải RealSense) được hiệu chuẩn luôn nội tham số từ chính các ảnh bảng: lúc đầu đưa bảng sát
webcam, nghiêng nhiều (tới ±45°) và phủ các góc ảnh, chỉ webcam cần thấy. Lần hiệu chuẩn sau tự dùng lại nội tham
số webcam đã lưu (cùng độ phân giải); thêm --redo-intrinsics để làm lại.
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
from openarm_shadow.cameras.calibration import (average_extrinsics, board_pose, calibrate_intrinsics, detect, make_board,
                                        relative_extrinsic, reprojection_rms_px)
from openarm_shadow.config import ROOT, load_config
from openarm_shadow.cameras import MultiCameraSource
from openarm_shadow.fusion import CameraModel
from openarm_shadow.viz import put_lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", default=None)
    ap.add_argument("--samples", type=int, default=20, help="số lần chụp (mọi camera cùng thấy bảng)")
    ap.add_argument("--min-corners", type=int, default=10)
    ap.add_argument("--intr-images", type=int, default=20, help="số ảnh riêng cho nội tham số mỗi webcam")
    ap.add_argument("--redo-intrinsics", action="store_true", help="không dùng lại nội tham số webcam đã lưu")
    args = ap.parse_args()
    cfg = load_config(args.config)
    fc = cfg["fusion"]
    if len(fc["cameras"]) < 2:
        raise SystemExit("fusion.cameras cần ít nhất 2 camera (xem config/fusion_2cam.yaml)")
    names = [c["name"] for c in fc["cameras"]]
    board, det = make_board(fc["board"])
    calib_path = Path(fc["calib_file"])
    calib_path = calib_path if calib_path.is_absolute() else ROOT / calib_path
    old = {}
    if calib_path.is_file() and not args.redo_intrinsics:
        old = (yaml.safe_load(calib_path.read_text()) or {}).get("cameras", {})
    src = MultiCameraSource(cfg)
    src.pair_wait = src.tol          # hiệu chuẩn cần khung chụp cùng lúc: luôn chờ khung khớp
    # Chỉ chụp khi bảng ĐỨNG YÊN (xem `still`), nên khung lệch vài chục ms vẫn dùng được: camera chậm/không đều
    # (điện thoại qua Wi-Fi ~20 fps) không bị bỏ. Lệch hơn max_lag_s thì bỏ riêng camera đó ở lần chụp này.
    max_lag_s = 0.1
    src.stale_s = max(src.stale_s, max_lag_s)
    prev_ref = None
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
                first = sizes[v] is None
                sizes[v] = (s.bgr.shape[1], s.bgr.shape[0])
                if cam.rs_intr is None and s.intrinsics is not None:
                    cam.set_realsense_intrinsics(s.intrinsics)
                o = old.get(names[v], {})
                if first and cam.rs_intr is None and o.get("K") and tuple(o.get("size") or ()) == sizes[v]:
                    cam.K, cam.dist, cam.size = np.asarray(o["K"], float), np.asarray(o["dist"], float), sizes[v]
                    print(f"{names[v]}: dùng lại nội tham số webcam đã lưu ({sizes[v][0]}x{sizes[v][1]})")
            dets = [detect(det, s.bgr) for s in ms.views]
            counts = [0 if ids is None else len(ids) for ids, _ in dets]
            now = time.monotonic()
            # Camera tham chiếu + ít nhất 1 camera phụ cùng thấy bảng (ngoại tham số tính theo từng cặp với camera 0).
            # Camera phụ có khung lệch quá max_lag_s / treo thì không tính ở lần này (các camera khác vẫn chụp).
            lags = ms.lags or [0.0] * len(cams)
            usable = [v == 0 or (not (ms.stale or [False] * len(cams))[v] and lags[v] <= max_lag_s)
                      for v in range(len(cams))]
            seen = tuple(c >= args.min_corners and u for c, u in zip(counts, usable))
            ref_center = None if dets[0][1] is None else dets[0][1].mean(axis=0)
            # Bảng đứng yên: tâm các góc ở camera tham chiếu xê dịch < 3 px so với khung trước (ở 30 fps ~ < 9 cm/s)
            still = ref_center is not None and prev_ref is not None and np.linalg.norm(ref_center - prev_ref) < 3.0
            prev_ref = ref_center
            all_ok = seen[0] and any(seen[1:]) and still
            moved = last_ref is None or (ref_center is not None and np.linalg.norm(ref_center - last_ref[0]) > 40) \
                or seen != last_ref[1]
            if all_ok and moved and now - last_t > 0.8:
                samples.append([d if ok_v else (None, None) for d, ok_v in zip(dets, seen)])
                last_t, last_ref = now, (ref_center, seen)
            pair_n = [sum(sm[v][0] is not None for sm in samples) for v in range(len(cams))]
            need_intr = [cams[v].rs_intr is None and cams[v].K is None for v in range(len(cams))]
            for v, (ids, px) in enumerate(dets):
                if not need_intr[v] or ids is None or len(ids) < 12 or len(intr_dets[v]) >= 3 * args.intr_images:
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
                put_lines(img, [f"{names[v]}: {counts[v]} goc" + ("" if v == 0 else f", lech {1000 * lags[v]:.0f} ms"
                                + ("" if usable[v] else " (BO)"))], org=(10, img.shape[0] - 14))
                tiles.append(cv2.resize(img, (int(img.shape[1] * 360 / img.shape[0]), 360)))
            view = np.hstack(tiles)
            lines = ["da chup " + " | ".join(f"{names[v]} {pair_n[v]}/{args.samples}" for v in range(1, len(cams)))
                     + f" | lech khung {1000 * ms.skew_s:.0f} ms",
                     f"cam bang de {names[0]} + camera khac cung thay, GIU YEN 1 giay, doi vi tri/goc nghieng"
                     " | c: tinh ngay | q: thoat"]
            for v in range(len(cams)):
                if need_intr[v]:
                    lines.append(f"{names[v]} noi tham so: {len(intr_dets[v])}/{args.intr_images} anh "
                                 "(dua bang sat webcam, NGHIENG NHIEU, phu cac goc anh)")
            put_lines(view, lines)
            cv2.imshow("calibrate", view)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                raise SystemExit("Huỷ, không lưu.")
            intr_ok = all(not need_intr[v] or len(intr_dets[v]) >= args.intr_images for v in range(len(cams)))
            if (min(pair_n[1:]) >= args.samples and intr_ok) or (k == ord("c") and min(pair_n[1:]) >= 8):
                break
    finally:
        src.close()
        cv2.destroyAllWindows()

    out = {"reference": names[0], "board": fc["board"],
           "created": datetime.datetime.now().isoformat(timespec="seconds"), "cameras": {}}
    for v, cam in enumerate(cams):
        if cam.rs_intr is None and cam.K is None:
            K, dist, rms = calibrate_intrinsics(board, intr_dets[v] + [s[v] for s in samples], sizes[v])
            cam.K, cam.dist, cam.size = K, dist, sizes[v]
            fov = np.degrees(2 * np.arctan(sizes[v][0] / (2 * K[0, 0])))
            print(f"{names[v]}: nội tham số webcam từ {len(intr_dets[v])} ảnh, sai số {rms:.2f} px, "
                  f"góc nhìn ngang {fov:.0f}°" + ("  <-- sai số lớn, nên làm lại" if rms > 1.0 else ""))

    def intr_entry(cam, v):
        if cam.rs_intr is not None:
            return {}
        return {"K": cam.K.tolist(), "dist": np.asarray(cam.dist).reshape(-1).tolist(), "size": list(sizes[v])}

    out["cameras"][names[0]] = {"R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0], **intr_entry(cams[0], 0)}
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
        rms = reprojection_rms_px(board, cams[0], cams[v], [(s[0], s[v]) for s in samples if s[v][0] is not None])
        ang = np.degrees(np.arccos(np.clip(cams[0].R[2] @ R[2], -1, 1)))
        base = np.linalg.norm(cams[v].center - cams[0].center)
        entry = {"R": R.tolist(), "t": t.tolist(), "rms_px": round(rms, 2), "samples_used": int(keep.sum())}
        entry.update(intr_entry(cams[v], v))
        out["cameras"][names[v]] = entry
        verdict = "tốt" if rms < 3 else ("tạm được" if rms < 6 else "KÉM, nên chụp lại")
        print(f"{names[v]}: lệch trục nhìn {ang:.1f}°, cách camera {names[0]} {base * 100:.0f} cm, "
              f"sai số chiếu lại {rms:.2f} px ({verdict}), dùng {int(keep.sum())}/{len(ests)} lần chụp")
    calib_path.write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True))
    print("Đã ghi", calib_path)


if __name__ == "__main__":
    main()
