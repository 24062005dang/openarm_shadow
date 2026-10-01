#!/usr/bin/env python3
"""Đo độ trễ thật từ một lần chạy có ghi (--record), để so trước/sau khi chỉnh bộ lọc, tốc độ, velocity_tracking.

    python scripts/shadow.py ... --record run1.npz      # cử động tay vài chục giây, đủ mọi khớp
    python scripts/measure_lag.py run1.npz

Với mỗi khớp có cử động (> 3°), in:
- mục tiêu -> lệnh : trễ do SafetyGate (giới hạn tốc độ, tăng tốc mềm, bám vận tốc)
- lệnh -> đo       : trễ do motor bám lệnh (kp/kd, quán tính, có/không feedforward vận tốc)
- mục tiêu -> đo   : tổng (chưa gồm trễ camera + nhận diện + bộ lọc trước mục tiêu)
Trễ = dịch thời gian làm hai tín hiệu khớp nhau nhất (tìm 0-600 ms, bước 5 ms). Robot mô phỏng: đo = lệnh.
"""
import argparse

import numpy as np

NAMES = ["J1", "J2", "J3", "J4", "J5", "J6", "J7"]


def best_lag(t, a, b, max_s=0.6, step=0.005):
    """Độ trễ (s) của b so với a: b(t) ~ a(t - lag). Trả (lag, sai số RMS sau khi dịch, độ)."""
    grid = np.arange(t[0] + max_s, t[-1], step)
    if len(grid) < 20:
        return np.nan, np.nan
    bb = np.interp(grid, t, b)
    best = (np.inf, np.nan)
    for lag in np.arange(0.0, max_s, step):
        e = np.sqrt(np.mean((np.interp(grid - lag, t, a) - bb) ** 2))
        if e < best[0]:
            best = (e, lag)
    return best[1], np.degrees(best[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    args = ap.parse_args()
    d = np.load(args.npz)
    if "ctl_t" not in d:
        raise SystemExit("File không có dữ liệu 100 Hz (ctl_*). Ghi lại bằng bản code mới: shadow.py ... --record x.npz")
    tc = d["ctl_t"]
    sides = [k[len("ctl_cmd_"):] for k in d.files if k.startswith("ctl_cmd_")]
    t_tg = d["t"]
    for s in sides:
        cmd, meas = d[f"ctl_cmd_{s}"], d[f"ctl_meas_{s}"]
        tgt = d[f"target_{s}"] if f"target_{s}" in d else None
        print(f"\nTay {s} ({len(tc)} nhịp điều khiển, {tc[-1] - tc[0]:.0f} s)")
        print(f"  {'khớp':4s} {'biên độ':>8s} {'mục tiêu->lệnh':>16s} {'lệnh->đo':>14s} {'mục tiêu->đo':>16s}")
        for j, n in enumerate(NAMES):
            if np.degrees(np.ptp(cmd[:, j])) < 3:
                continue
            l_cm, e_cm = best_lag(tc, cmd[:, j], meas[:, j])
            row = f"  {n:4s} {np.degrees(np.ptp(cmd[:, j])):7.0f}° {'':>16s} {1000 * l_cm:8.0f} ms ({e_cm:3.1f}°)"
            if tgt is not None:
                ok = np.isfinite(tgt[:, j])
                if ok.sum() > 20:
                    tg = np.interp(tc, t_tg[ok], tgt[ok, j])
                    l_tc, _ = best_lag(tc, tg, cmd[:, j])
                    l_tm, e_tm = best_lag(tc, tg, meas[:, j])
                    row = (f"  {n:4s} {np.degrees(np.ptp(cmd[:, j])):7.0f}° {1000 * l_tc:13.0f} ms "
                           f"{1000 * l_cm:8.0f} ms ({e_cm:3.1f}°) {1000 * l_tm:10.0f} ms ({e_tm:3.1f}°)")
            print(row)
    print("\n(số trong ngoặc: sai số RMS còn lại sau khi bù trễ - lớn nghĩa là robot không theo kịp biên độ, "
          "vd do giới hạn tốc độ hoặc trọng lực)")


if __name__ == "__main__":
    main()
