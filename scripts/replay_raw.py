#!/usr/bin/env python3
"""Chạy lại perception + fusion + pipeline trên dữ liệu thô đã ghi (scripts/record_multicam_raw.py), không cần
camera hay robot. Dùng để thử thay đổi perception/bộ lọc và để kiểm tra hồi quy: cùng dữ liệu, cùng config thì đầu ra
phải giống hệt trước và sau khi sửa code (so bằng --compare).

    python scripts/replay_raw.py data/2026-10-03_raw_bimanual_take2 \
        --intrinsics data/2026-10-03_raw_bimanual_take2/intrinsics_approx.yaml --frames 600 -o replay.npz
    python scripts/replay_raw.py ... -o new.npz --compare replay.npz

Mặc định dùng effective_config.yaml lưu trong thư mục ghi (đúng config lúc ghi); --config để ghép đè thêm.
Hiệu chuẩn camera: cameras_calib.yaml trong thư mục ghi. Hướng tay tự hiệu chuẩn như lúc chạy (tay buông, xoè,
lòng bàn tay nhìn camera); xong cả hai tay thì coi như đã engage và không hiệu chuẩn lại.

File .npz: t, và cho mỗi tay robot: target_*, conf_*, held_*, fus_* (như --record), obs_* (vai, khuỷu, cổ tay
khung thân, 9 số), H_* (hướng bàn tay 9 số), grip_* (tỉ số ngón cái-trỏ), calib_* (đã hiệu chuẩn hướng tay),
body_est (orientation.body_ref bật: số điểm vai/hông đang dùng thân chuẩn vì bị che; NaN khi chưa học xong).
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from openarm_shadow.cameras.raw_replay import RawReplaySource, load_intrinsics  # noqa: E402
from openarm_shadow.config import _merge, resolve_model_paths  # noqa: E402
from openarm_shadow.fusion import MultiViewPerception  # noqa: E402
from openarm_shadow.retarget.pipeline import ShadowPipeline  # noqa: E402
from openarm_shadow.runtime.recorder import fusion_row  # noqa: E402


def nan_vec(n):
    return np.full(n, np.nan)


def compare(a_path, b_path):
    """In sai khác lớn nhất từng khoá giữa hai file replay. -> True nếu giống hệt (bỏ qua NaN cùng chỗ)."""
    a, b = np.load(a_path), np.load(b_path)
    same = True
    for k in sorted(set(a.files) | set(b.files)):
        if k not in a.files or k not in b.files:
            print(f"  {k}: chỉ có ở {'B' if k not in a.files else 'A'}")
            same = False
            continue
        x, y = a[k].astype(float), b[k].astype(float)
        if x.shape != y.shape:
            print(f"  {k}: khác kích thước {x.shape} / {y.shape}")
            same = False
            continue
        nan_diff = int(np.count_nonzero(np.isnan(x) != np.isnan(y)))
        both = ~np.isnan(x) & ~np.isnan(y)
        d = float(np.max(np.abs(x[both] - y[both]))) if both.any() else 0.0
        if nan_diff or d > 0:
            same = False
            print(f"  {k}: lệch lớn nhất {d:.3g}, khác chỗ NaN {nan_diff}")
    print("GIỐNG HỆT" if same else "CÓ KHÁC BIỆT")
    return same


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="thư mục ghi thô")
    ap.add_argument("--config", action="append", default=None, help="ghép đè lên effective_config.yaml")
    ap.add_argument("--intrinsics", default=None, help="nội tham số RealSense nếu thư mục ghi không có intrinsics.yaml")
    ap.add_argument("--start", type=float, default=0.0, help="bỏ qua chừng này giây đầu")
    ap.add_argument("--frames", type=int, default=None, help="số khung tham chiếu tối đa")
    ap.add_argument("--stride", type=int, default=1, help="dùng 1/stride khung tham chiếu")
    ap.add_argument("-o", "--out", default="replay.npz")
    ap.add_argument("--compare", default=None, help="so đầu ra với file replay khác")
    args = ap.parse_args()

    root = Path(args.root)
    cfg = yaml.safe_load((root / "effective_config.yaml").read_text())
    for p in args.config or []:
        cfg = _merge(cfg, yaml.safe_load(Path(p).read_text()))
    resolve_model_paths(cfg)
    fc = cfg["fusion"]
    if (root / Path(fc["calib_file"]).name).is_file():
        fc["calib_file"] = str((root / Path(fc["calib_file"]).name).resolve())
    names = [c["name"] for c in fc["cameras"]]
    src = RawReplaySource(root, names, [float(c.get("latency_s", 0.0)) for c in fc["cameras"]],
                          float(fc.get("max_skew_s", 0.04)),
                          load_intrinsics(args.intrinsics) if args.intrinsics else None,
                          args.start, args.frames, args.stride)
    ok, first = src.read()
    if not ok:
        raise SystemExit("Không đọc được khung đầu: " + str(src.error))
    perc = MultiViewPerception.from_config(cfg, first)
    pipe = ShadowPipeline(cfg)
    pipe.seed({s: np.zeros(8) for s in pipe.robot_sides})
    sides = pipe.robot_sides
    log = {"t": []}
    for s in sides:
        for k in ("target", "conf", "held", "fus", "obs", "H", "grip", "calib"):
            log[f"{k}_{s}"] = []
    engaged = False
    t_start, n = time.monotonic(), 0
    sample = first
    try:
        while True:
            fr = perc.process(sample)
            if not engaged:
                pipe.auto_calibrate_hand_neutral(fr)
                engaged = perc.body_ready() and all(pipe.hand_calibrated[s] for s in sides)
            targets = pipe.step(fr)
            log["t"].append(fr.t)
            if perc.body_ref.enabled:            # orientation.body_ref: số điểm vai/hông đang ước lượng (bị che)
                log.setdefault("body_est", []).append(sum(w < 0.5 for w in perc.body_ref.weights.values())
                                                      if perc.body_ref.ready else np.nan)
            for s in sides:
                ob = pipe._obs_for_robot(fr, s)
                log[f"target_{s}"].append(targets[s])
                log[f"conf_{s}"].append(pipe.conf[s])
                log[f"held_{s}"].append(pipe.held[s].astype(float))
                log[f"fus_{s}"].append(fusion_row(fr, pipe.human_side_for(s)))
                log[f"obs_{s}"].append(np.concatenate([nan_vec(3) if v is None else v for v in (ob.s, ob.e, ob.w)]))
                log[f"H_{s}"].append(nan_vec(9) if ob.H is None else np.asarray(ob.H).ravel())
                log[f"grip_{s}"].append(np.nan if ob.grip is None else ob.grip)
                log[f"calib_{s}"].append(float(pipe.hand_calibrated[s]))
            n += 1
            if n % 100 == 0:
                print(f"  {n} khung, {n / (time.monotonic() - t_start):.1f} khung/s", flush=True)
            ok, sample = src.read()
            if not ok:
                break
    finally:
        perc.close()
        src.close()
    np.savez(args.out, **{k: np.asarray(v) for k, v in log.items()})
    print(f"Đã lưu {args.out}: {n} khung, {n / max(time.monotonic() - t_start, 1e-6):.1f} khung/s")
    for s in sides:
        tg = np.asarray(log[f"target_{s}"])
        print(f"  {s}: có mục tiêu J1-J7 {100 * np.mean(np.isfinite(tg[:, :7])):.0f}%, "
              f"hiệu chuẩn tay {'xong' if log[f'calib_{s}'] and log[f'calib_{s}'][-1] else 'CHƯA'}")
    if args.compare:
        print(f"So với {args.compare}:")
        if not compare(args.compare, args.out):
            sys.exit(1)


if __name__ == "__main__":
    main()
