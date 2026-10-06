#!/usr/bin/env python3
"""So sánh nhiều lần chạy (file --record) cùng động tác: độ "động" và số lần nhảy của góc mục tiêu từng khớp.

    python scripts/compare_records.py off.npz body_ref.npz kalman.npz both.npz

Quãng đường góc mỗi giây: đứng yên mà số lớn = rung; cùng động tác thì số nhỏ hơn = êm hơn (nhưng quá nhỏ có thể là
đang bỏ cử động thật). Nhảy > 10° / > 20° giữa 2 khung liên tiếp: giật. Chỉ tính từ khung đầu tiên có mục tiêu.
"""
import sys
from pathlib import Path

import numpy as np


def stats(path):
    d = np.load(path)
    t = d["t"]
    out = {}
    for s in ("right", "left"):
        if f"target_{s}" not in d.files:
            continue
        tg = np.degrees(d[f"target_{s}"][:, :7])
        ok = np.flatnonzero(np.all(np.isfinite(tg[:, :4]), axis=1))
        if not len(ok):
            continue
        tg = tg[ok[0]:]
        dur = max(t[-1] - t[ok[0]], 1e-6)
        dd = np.abs(np.diff(tg, axis=0))
        out[s] = {"J1-J3 °/s": np.nansum(dd[:, :3]) / dur, "J4-J7 °/s": np.nansum(dd[:, 3:]) / dur,
                  "nhảy>10°": int(np.nansum(dd > 10)), "nhảy>20°": int(np.nansum(dd > 20)), "thời gian (s)": dur}
    return out


runs = {Path(p).stem: stats(p) for p in sys.argv[1:]}
names = list(runs)
print(f"{'':24s}" + "".join(f"{n:>14s}" for n in names))
for s in ("right", "left"):
    keys = next((list(r[s]) for r in runs.values() if s in r), [])
    for k in keys:
        vals = [runs[n].get(s, {}).get(k, float("nan")) for n in names]
        fmt = "{:14.0f}" if isinstance(vals[0], int) or k.startswith("thời") else "{:14.1f}"
        print(f"{s + ' ' + k:24s}" + "".join(fmt.format(v) for v in vals))
