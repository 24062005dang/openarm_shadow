"""Động học một tay OpenArm v1.0, đọc từ data/openarm_v10_arms.json (sinh từ URDF).

Khung gốc (khung 0) = khung world của URDF: x ra trước, y sang trái, z lên.
Đã kiểm tra bằng FK: J4 = +90° đưa cẳng tay ra +x; tay phải nằm phía −y.
Góc khớp ở đây là góc **theo URDF**; đổi sang góc motor nằm ở robot/mapping.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import rot, rpy_to_matrix, unit

DATA = Path(__file__).parent / "data" / "openarm_v10_arms.json"


@dataclass
class Joint:
    name: str
    R_local: np.ndarray   # quay cố định từ khung trước tới khung khớp (origin rpy)
    p_local: np.ndarray   # dịch cố định (origin xyz)
    axis: np.ndarray      # trục quay trong khung khớp
    lower: float
    upper: float


class ArmKinematics:
    def __init__(self, side: str, data_path: Path = DATA):
        assert side in ("right", "left")
        self.side = side
        arm = json.loads(Path(data_path).read_text())["arms"][side]
        self.R_base = rpy_to_matrix(*arm["base"]["rpy"])
        self.p_base = np.array(arm["base"]["xyz"])
        self.joints = [
            Joint(j["name"], rpy_to_matrix(*j["rpy"]), np.array(j["xyz"]), unit(j["axis"]),
                  j["lower"], j["upper"])
            for j in arm["joints"]
        ]
        self.R_tcp = rpy_to_matrix(*arm["tcp"]["rpy"])
        self.p_tcp = np.array(arm["tcp"]["xyz"])
        self.p_finger = np.array(arm["finger_tip"]["xyz"])
        self.lower = np.array([j.lower for j in self.joints])
        self.upper = np.array([j.upper for j in self.joints])
        # dấu để "trục khớp i" cùng chiều với đoạn chi (vai→khuỷu, khuỷu→cổ tay) ở q = 0
        q0 = np.zeros(7)
        P = self.joint_positions(q0)
        self.limb_sign = {
            3: float(np.sign(self.axis_world(q0, 3) @ (P[4] - P[2]))) or 1.0,
            5: float(np.sign(self.axis_world(q0, 5) @ (P[6] - P[4]))) or 1.0,
        }

    # ---------- FK ----------
    def frames(self, q):
        """Danh sách (R, p) của khung 0..7 trong world. Khung 0 = link0 của tay."""
        R, p = self.R_base.copy(), self.p_base.copy()
        out = [(R, p)]
        for j, qi in zip(self.joints, q):
            p = p + R @ j.p_local
            R = R @ j.R_local @ rot(j.axis, qi)
            out.append((R, p))
        return out

    def R0(self, q, i):
        """Hướng khung i (0..7) trong world."""
        return self.frames(q)[i][0]

    def axis_world(self, q, i):
        """Trục khớp i (1..7) trong world (không phụ thuộc q_i)."""
        return self.R0(q, i) @ self.joints[i - 1].axis

    def joint_positions(self, q):
        """Vị trí gốc khớp 1..7 + TCP + đầu ngón, dạng mảng (9, 3). P[0] = gốc khớp 1."""
        fr = self.frames(q)
        pts = [fr[i][1] for i in range(1, 8)]
        R7, p7 = fr[7]
        pts.append(p7 + R7 @ self.p_tcp)
        pts.append(p7 + R7 @ self.p_finger)
        return np.array(pts)

    def keypoints(self, q):
        """Vai, khuỷu, cổ tay, đầu kẹp của robot (dùng cho vẽ và kiểm tra va chạm)."""
        P = self.joint_positions(q)
        return {"shoulder": P[1], "elbow": P[3], "wrist": P[5], "tool": P[8]}

    def R_tool(self, q):
        return self.R0(q, 7) @ self.R_tcp

    def clamp(self, q):
        return np.clip(q, self.lower, self.upper)
