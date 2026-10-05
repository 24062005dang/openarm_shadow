#!/usr/bin/env python3
"""Phân tích các lệnh node teleop đã phát qua ROS 2 (file .npz của scripts/fake_openarm_backend.py), để kiểm tra trước
khi gửi bên backend: tần số, quỹ đạo có nhảy không, tốc độ / gia tốc, giới hạn khớp, mức kẹp, robot theo kịp không.

    python scripts/analyze_ros2_log.py backend.npz [--record shadow.npz] [--plot ros2_lenh.png] [--levels 10]

--record: file --record của chính node (mục tiêu trước SafetyGate) để đếm các lần MỤC TIÊU nhảy lớn (perception) mà
SafetyGate đã làm mượt. Thoát mã 1 nếu có lỗi định dạng / vượt giới hạn khớp / bước lệnh vượt ngưỡng --max-step-deg.
"""
import argparse
import sys

import numpy as np

ARM = [f"{s}_j{i}" for s in ("left", "right") for i in range(1, 8)]
ARM_IDX = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14]
LIMITS = np.array([(-3.4907, 1.3963), (-3.3161, 0.17453), (-1.5708, 1.5708), (0.0, 2.4435), (-1.5708, 1.5708),
                   (-0.7854, 0.7854), (-1.5708, 1.5708),
                   (-1.3963, 3.4907), (-0.17453, 3.3161), (-1.5708, 1.5708), (0.0, 2.4435), (-1.5708, 1.5708),
                   (-0.7854, 0.7854), (-1.5708, 1.5708)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--record", default=None)
    ap.add_argument("--plot", default=None)
    ap.add_argument("--levels", type=int, default=10, help="số mức kẹp mong đợi (0 = liên tục)")
    ap.add_argument("--max-step-deg", type=float, default=2.0, help="bước lệnh lớn nhất cho phép giữa 2 lệnh liên tiếp")
    a = ap.parse_args()
    d = np.load(a.log)
    t, arm = d["arm_t"], d["arm"]
    ok = True
    print(f"== Lệnh khớp: {len(t)} lệnh, bỏ vì sai số lượng: {int(d['rejected'])}")
    if len(t) < 10:
        print("   quá ít lệnh để phân tích (node đã engage chưa?)")
        sys.exit(1)
    gaps = np.diff(t) * 1000
    print(f"   từ {t[0]:.1f} s tới {t[-1]:.1f} s, tần số {(len(t) - 1) / (t[-1] - t[0]):.1f} Hz, khoảng cách "
          f"trung vị {np.median(gaps):.1f} ms, p95 {np.percentile(gaps, 95):.1f} ms, lớn nhất {gaps.max():.1f} ms")
    bad = int(np.count_nonzero(~np.isfinite(arm)))
    lo, hi = LIMITS[:, 0], LIMITS[:, 1]
    out = int(np.count_nonzero((arm < lo - 1e-6) | (arm > hi + 1e-6)))
    print(f"   NaN/inf: {bad} | ngoài JOINT_LIMITS backend: {out}")
    ok &= bad == 0 and out == 0

    dq = np.diff(arm, axis=0)
    dts = np.diff(t)[:, None]
    vel = np.degrees(dq / np.maximum(dts, 1e-3))
    acc = np.diff(vel, axis=0) / np.maximum(np.diff(t)[1:, None], 1e-3)
    step = np.degrees(np.abs(dq))
    print(f"\n   {'khớp':9s} {'dải (°)':>16s} {'bước max (°)':>13s} {'v max (°/s)':>12s} {'v p99':>7s} "
          f"{'a p99 (°/s²)':>13s} {'bước >' + str(a.max_step_deg) + '°':>10s}")
    for k, n in enumerate(ARM):
        q = np.degrees(arm[:, k])
        nbig = int(np.count_nonzero(step[:, k] > a.max_step_deg))
        ok &= nbig == 0
        print(f"   {n:9s} {q.min():7.1f}..{q.max():6.1f} {step[:, k].max():13.2f} {np.abs(vel[:, k]).max():12.1f} "
              f"{np.percentile(np.abs(vel[:, k]), 99):7.1f} {np.percentile(np.abs(acc[:, k]), 99):13.0f} {nbig:10d}")

    st_t, st = d["state_t"], d["state"]
    if len(st_t):
        s_at = np.array([np.interp(t, st_t, st[:, i]) for i in ARM_IDX]).T
        lag = np.degrees(np.abs(arm - s_at))
        print(f"\n== Robot (giả, giới hạn {float(d['velocity_limit']):.2f} rad/s) cách lệnh: trung vị "
              f"{np.median(lag):.1f}°, lớn nhất {lag.max():.1f}° ({ARM[int(np.argmax(lag.max(0)))]})")
        first = np.degrees(np.abs(arm[0] - s_at[0])).max()
        print(f"   lệnh đầu tiên lệch tư thế đo {first:.2f}° (đồng bộ khi engage: phải gần 0)")

    gt, gs, gv = d["grip_t"], d["grip_side"], d["grip"]
    print(f"\n== Kẹp: {len(gt)} lệnh")
    for side, name in ((0, "trái"), (1, "phải")):
        v = gv[gs == side]
        if not len(v):
            print(f"   {name}: không có lệnh")
            continue
        vals = sorted(set(np.round(v * 1000, 2)))
        span = gt[gs == side][-1] - gt[gs == side][0] if len(v) > 1 else 0.0
        print(f"   {name}: {len(v)} lệnh ({len(v) / max(span, 1e-6):.1f} lệnh/s), {len(vals)} giá trị khác nhau (mm): "
              + ", ".join(f"{x:g}" for x in vals))
        if a.levels >= 2:
            allowed = np.round(np.linspace(0, 43.0, a.levels), 2)
            off = [x for x in vals if np.min(np.abs(allowed - x)) > 0.01]
            print(f"   {name}: ngoài {a.levels} mức cho phép: {off or 'không'}")
            ok &= not off

    if a.record:
        r = np.load(a.record)
        print("\n== Mục tiêu trước SafetyGate (--record): số lần nhảy > 20°/khung (perception), đã được làm mượt")
        for s in ("left", "right"):
            if f"target_{s}" not in r.files:
                continue
            tg = np.degrees(r[f"target_{s}"][:, :7])
            jumps = np.nansum(np.abs(np.diff(tg, axis=0)) > 20, axis=0).astype(int)
            print(f"   {s}: " + " ".join(f"J{i + 1}:{j}" for i, j in enumerate(jumps)))

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(8, 2, figsize=(14, 18), sharex=True)
        for k, n in enumerate(ARM):
            ax = axs[k % 7, k // 7]
            ax.plot(t, np.degrees(arm[:, k]), lw=1, label="lệnh (đã phát)")
            if len(st_t):
                ax.plot(st_t, np.degrees(st[:, ARM_IDX[k]]), lw=1, ls="--", label="robot giả")
            ax.set_ylabel(f"{n} (°)")
            ax.grid(alpha=0.3)
        for side, col in ((0, 0), (1, 1)):
            ax = axs[7, col]
            m = gs == side
            ax.step(gt[m], gv[m] * 1000, where="post", lw=1, label="lệnh kẹp")
            if len(st_t):
                ax.plot(st_t, st[:, 7 if side == 0 else 15] * 1000, lw=1, ls="--", label="kẹp giả")
            ax.set_ylabel(("left" if side == 0 else "right") + "_gripper (mm)")
            ax.set_xlabel("thời gian (s)")
            ax.grid(alpha=0.3)
        axs[0, 0].legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(a.plot, dpi=90)
        print(f"\nĐồ thị: {a.plot}")
    print("\nKẾT LUẬN:", "ĐẠT" if ok else "CÓ VẤN ĐỀ (xem các dòng trên)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
