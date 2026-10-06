"""Dòng chẩn đoán fusion hiện trên màn hình."""
from __future__ import annotations

import numpy as np


def fusion_lines(fr, sides):
    """Dòng chẩn đoán fusion: số camera thấy vai/khuỷu/cổ tay, sai số chiếu lại, xung đột depth, bàn tay."""
    fi = fr.fusion or {}
    people = fi.get("people")
    out = [f"fusion {fi.get('views', 0)} cam | lech khung {fi.get('skew_ms', 0.0):.0f} ms"
           + (" | nguoi thay: " + "/".join(str(n) for n in people) + " (khoa 1 nguoi)" if people else "")
           + (f" | MAT KHUNG: {', '.join(fi['stale'])}" if fi.get("stale") else "")]
    match = fi.get("person") or {}
    if match:
        # Camera phụ có chọn đúng người camera 0 đang khoá không (multiview._match_operator)
        def txt(m):
            if m["mode"] == "tu khoa":
                return "tu khoa"
            err = m.get("err", float("nan"))
            val = "" if not np.isfinite(err) else (f" {err:.2f}" if m["mode"] == "3D" else f" {err:.0f}px")
            return ("khop " if m["ok"] else "KHONG KHOP ") + m["mode"] + val
        out.append("cung 1 nguoi: " + " | ".join(f"{n} {txt(m)}" for n, m in match.items()))
    names = {"right": (12, 14, 16), "left": (11, 13, 15)}
    for s in sides:
        pts = fi.get("points", {})
        if fi.get("body") == "front":
            out.append(f"{s}: vai/khuyu/co tay tu Pose camera 0")
            pts = None
        parts = []
        for tag, i in zip(("vai", "khuyu", "co tay"), names[s] if pts is not None else ()):
            p = pts.get(i)
            if p is None:
                parts.append(f"{tag} -")
                continue
            err = p.get("err_px", float("nan"))
            parts.append(f"{tag} {p['views']}cam" + (f" {err:.0f}px" if np.isfinite(err) else "") +
                         (" D" if p.get("depth") else "") + (" !" if p.get("conflict") else ""))
        if parts:
            out.append(f"{s}: " + " | ".join(parts))
        h = fi.get(f"hand_{s}")
        if h:
            extra = ""
            if h.get("fit") == "KABSCH" and np.isfinite(h.get("fit_mm", np.nan)):
                extra += f", khop long tay {h['fit_mm']:.0f}mm"
            elif h.get("fit") == "3PT":
                extra += ", dang hoc khuon long tay"
            if h.get("rejected"):
                extra += f", bo {h['rejected']} cam (xa co tay)"
            if h.get("bones_dropped"):
                extra += f", bo {h['bones_dropped']} diem (dot bat thuong)"
            if h.get("sources"):
                extra += " | nguon: " + "+".join(h["sources"])
            out.append(f"  ban tay {h['views']}cam {h['points']}/21 diem, nhin ro {h['quality']:.2f}, "
                       f"{h['mode']}{extra}")
    return out
