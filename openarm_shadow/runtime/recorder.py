"""Ghi --record: mục tiêu, lệnh, độ tin cậy, chẩn đoán fusion mỗi khung; lệnh và góc đo mỗi nhịp điều khiển."""
from __future__ import annotations

import numpy as np

from ..perception.types import ARM_IDX


def fusion_row(fr, human_side):
    """Chẩn đoán hợp nhất vai/khuỷu/cổ tay cho --record: [số camera x3, sai số chiếu lại px x3, depth x3] (NaN nếu
    không có, vd chế độ front hoặc 1 camera)."""
    pts = (getattr(fr, "fusion", None) or {}).get("points", {})
    row = np.full(9, np.nan)
    for k, i in enumerate(ARM_IDX[human_side]):
        p = pts.get(i)
        if p:
            row[k], row[3 + k], row[6 + k] = p.get("views", np.nan), p.get("err_px", np.nan), p.get("depth", np.nan)
    return row


class Recorder:
    """Gom dữ liệu --record trong bộ nhớ, lưu .npz khi kết thúc.

    Mỗi khung perception: t, target_<tay>, cmd_<tay>, conf_<tay>, fus_<tay> (fusion_row). Mỗi nhịp điều khiển
    (trace của runtime.controller.Controller): ctl_t, ctl_cmd_<tay>, ctl_meas_<tay>."""

    def __init__(self, sides):
        self.log = {"t": [], **{f"{k}_{s}": [] for k in ("target", "cmd", "conf", "fus") for s in sides}}
        self.trace = []

    def add(self, pipe, fr, targets, cmd):
        self.log["t"].append(fr.t)
        for s in pipe.robot_sides:
            self.log[f"target_{s}"].append(targets[s])
            self.log[f"cmd_{s}"].append(cmd[s])
            self.log[f"conf_{s}"].append(pipe.conf[s])
            self.log[f"fus_{s}"].append(fusion_row(fr, pipe.human_side_for(s)))

    def save(self, path):
        """Lưu nếu có ít nhất 1 khung. -> True nếu đã lưu."""
        if not self.log["t"]:
            return False
        log = dict(self.log)
        if self.trace:
            log["ctl_t"] = [x[0] for x in self.trace]
            for s in self.trace[0][1]:
                log[f"ctl_cmd_{s}"] = [x[1][s] for x in self.trace]
                log[f"ctl_meas_{s}"] = [x[2][s] for x in self.trace]
        np.savez(path, **{k: np.asarray(v) for k, v in log.items()})
        return True
