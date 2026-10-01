#!/usr/bin/env python3
"""Tìm các lần mục tiêu khớp nhảy lớn trong một lần chạy có ghi (--record), kèm chẩn đoán vì sao.

    python scripts/shadow.py ... --record run.npz
    python scripts/find_jumps.py run.npz              # ngưỡng mặc định 20°
    python scripts/find_jumps.py run.npz --deg 30

Mỗi dòng: thời điểm, khớp, góc trước -> sau, khớp đã giữ/mất bao lâu trước đó, cả 7 góc, và (file ghi bằng bản
mới) độ tin cậy khớp + vai/khuỷu/cổ tay do mấy camera thấy, sai số chiếu lại, có dùng depth không.
Đọc nhanh: nhảy ngay sau khi mất tay lâu (> 0,5 s) thường là cử động thật lúc không thấy tay; nhảy khi khuỷu chỉ
1 camera + depth (1c D) hoặc sai số lớn thường là lỗi nhận diện -> robot thật sẽ xoay tới tư thế sai.
"""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--deg", type=float, default=20.0)
    args = ap.parse_args()
    d = np.load(args.npz)
    t = d["t"] - d["t"][0]
    sides = [k[len("target_"):] for k in d.files if k.startswith("target_")]
    for s in sides:
        g = np.degrees(d[f"target_{s}"][:, :7])
        conf = d[f"conf_{s}"] if f"conf_{s}" in d else None
        fus = d[f"fus_{s}"] if f"fus_{s}" in d else None
        print(f"\nTay {s}: {len(t)} khung, {t[-1]:.0f} s")
        for j in range(7):
            ok = np.isfinite(g[:, j])
            print(f"  J{j + 1}: có mục tiêu {100 * ok.mean():3.0f}%", end="")
            if conf is not None:
                print(f" | độ tin cậy trung vị {np.median(conf[:, j]):.2f}", end="")
            print()
        if fus is not None and np.isfinite(fus[:, 1]).any():
            for k, n in enumerate(("vai", "khuỷu", "cổ tay")):
                v = fus[:, k]
                seen = np.isfinite(v) & (v > 0)
                if seen.any():
                    print(f"  {n:6s}: 2 camera {100 * np.mean(v[seen] >= 2):3.0f}% | 1 camera + depth "
                          f"{100 * np.mean(v[seen] == 1):3.0f}% | sai số trung vị {np.nanmedian(fus[seen, 3 + k]):4.1f} px")
        last_t, last_v, n_ev = np.full(7, np.nan), np.full(7, np.nan), 0
        for i in range(len(t)):
            for j in range(7):
                if not np.isfinite(g[i, j]):
                    continue
                if np.isfinite(last_v[j]) and abs(g[i, j] - last_v[j]) > args.deg:
                    n_ev += 1
                    line = (f"  t={t[i]:6.2f}s J{j + 1}: {last_v[j]:6.1f} -> {g[i, j]:6.1f}° "
                            f"(giữ/mất {1000 * (t[i] - last_t[j]):4.0f} ms) | " +
                            " ".join(f"{v:4.0f}" for v in g[i]))
                    if conf is not None:
                        line += f" | tin cậy {conf[i, j]:.2f}"
                    if fus is not None and np.isfinite(fus[i, 1]):
                        tags = []
                        for k, n in enumerate(("vai", "khuyu", "co tay")):
                            nv, e, dp = fus[i, k], fus[i, 3 + k], fus[i, 6 + k]
                            tags.append(f"{n} {int(nv)}c" + (f" {e:.0f}px" if np.isfinite(e) else "") +
                                        (" D" if dp > 0 else ""))
                        line += " | " + ", ".join(tags)
                    print(line)
                last_t[j], last_v[j] = t[i], g[i, j]
        print(f"  => {n_ev} lần nhảy > {args.deg:.0f}°/khung")


if __name__ == "__main__":
    main()
