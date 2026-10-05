"""Các dòng chữ chẩn đoán vẽ trên màn hình."""
from __future__ import annotations

import cv2
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
        # Camera phụ có chọn đúng người camera 0 đang khoá không (fusion.person_match)
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


def compact_fusion_lines(lines):
    """Chế độ màn hình compact: giữ dòng sync/mất khung và chất lượng hai bàn tay; bỏ phần khớp người/chi tiết
    từng vai-khuỷu-cổ tay vốn tạo rất nhiều chữ trên ba camera."""
    short, side = [lines[0]], None
    for ln in lines[1:]:
        if ln.startswith(("right:", "left:")):
            side = ln.split(":", 1)[0]
        elif ln.startswith("  ban tay"):
            short.append(f"{side or '?'} {ln.strip()}")
    return short


def draw_ready_badge(cam, ready, engaged, countdown=None):
    """Chấm tròn góc phải: viền cam = cần hiệu chuẩn, xanh lá = READY / đếm ngược tự engage, xanh dương = FOLLOW."""
    cx, cy = cam.shape[1] - 28, 28
    if ready and not engaged:
        cv2.circle(cam, (cx, cy), 15, (0, 220, 0), -1)
        text = f"AUTO SYNC {countdown:.1f}s" if countdown is not None else "READY"
        cv2.putText(cam, text, (max(8, cx - 175), cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    elif ready:
        cv2.circle(cam, (cx, cy), 15, (255, 200, 0), -1)
        cv2.putText(cam, "FOLLOW", (max(8, cx - 90), cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)
    else:
        cv2.circle(cam, (cx, cy), 15, (0, 180, 255), 2)
        cv2.putText(cam, "CALIB: ARM DOWN + PALM TO CAM", (max(8, cx - 285), cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 2)


def status_lines(head, pipe, fr, targets, cmd, multi, has_depth, compact):
    """Các dòng chữ trạng thái: head (fps/trạng thái, hướng dẫn phím), kẹp, fusion hoặc depth D455, hiệu chuẩn tay,
    sai số retarget, khung xương, mục tiêu/lệnh J1-J7 (bỏ các dòng chi tiết khi compact)."""
    lines = list(head)
    for s in pipe.robot_sides:
        gm = pipe.grip[s]
        rem = gm.calib_remaining(fr.t)
        gl = (f"{s} kep: r {gm.r:4.2f} -> {gm.value:4.2f} (chum {gm.pinch:.2f} / xoe {gm.open:.2f})"
              if np.isfinite(gm.r) else f"{s} kep: khong thay ngon cai/tro")
        if rem is not None:
            gl += f" | CALIB KEP {rem:3.1f}s: chum het co roi xoe het co"
        lines.append(gl)
    if multi:
        fl = fusion_lines(fr, pipe.robot_sides)
        lines += compact_fusion_lines(fl) if compact else fl
    elif has_depth:
        ds = " ".join(f"{s}:{fr.depth_used.get(s, 0)}/3" for s in pipe.robot_sides)
        lines.append("D455 depth vai/khuyu/co tay " + ds + " (3/3 = dang dung depth)")
        for human_side, di in fr.hand_depth.items():
            lines.append(f"{human_side} hand: {di['mode']} depth {di['direct']}/21 "
                         f"fused {di['fused']}/21 conf {di['confidence']:.2f}")
            rms = di.get("plane_rms_m", float("inf"))
            rms_text = f"{1000*rms:.1f}mm" if np.isfinite(rms) else "--"
            lines.append(f"  orient {di.get('orientation', 'NONE')} open {di.get('open_fingers', 0)}/4 "
                         f"plane {di.get('plane_inliers', 0)} rms {rms_text}")
    cs = " ".join(f"{s}:OK" if pipe.hand_calibrated[s] else
                  f"{s}:{100 * pipe.calib_progress[s]:.0f}% [{pipe.calib_hint[s]}]"
                  for s in pipe.robot_sides)
    lines.append("Auto calib tay: " + cs)
    if compact:
        return lines
    for s in pipe.robot_sides:
        inf = pipe.last_info.get(s)
        if inf is not None:
            lines.append(f"{s}: err u {inf.err_upper_deg:5.1f} l {inf.err_fore_deg:5.1f} "
                         f"tay {inf.err_hand_deg:5.1f} deg" + (" [thang]" if inf.elbow_straight else ""))
        ai = pipe.arm_shape_info.get(s)
        if ai is not None:
            # Lọc khung xương: độ dài đoạn tay khung này (cm) x tỉ lệ so với độ dài đã học; BO = khớp giữ
            def seg(name, k):
                r = ai["ratio"][k]
                return (f"{name} {100 * ai['len'][k]:.0f}cm" + (f" x{r:.2f}" if np.isfinite(r) else
                        " (dang hoc)") + ("" if ai["ok"][k] else " BO"))
            lines.append(f"{s} xuong: " + seg("tren", 0) + " | " + seg("cang", 1))
        if s in targets:
            at, ac = np.rad2deg(targets[s][:4]), np.rad2deg(cmd[s][:4])
            lines.append(f"{s} J1-4 target " + " ".join(f"{v:5.1f}" for v in at) + " | cmd " +
                         " ".join(f"{v:5.1f}" for v in ac))
        if s in targets and np.all(np.isfinite(targets[s][4:7])):
            wt = np.rad2deg(targets[s][4:7])
            wc = np.rad2deg(cmd[s][4:7])
            lines.append(f"{s} J5-7 target {wt[0]:5.1f} {wt[1]:5.1f} {wt[2]:5.1f} | "
                         f"cmd {wc[0]:5.1f} {wc[1]:5.1f} {wc[2]:5.1f}")
    return lines
