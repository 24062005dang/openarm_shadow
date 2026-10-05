#!/usr/bin/env python3
"""Sinh lại openarm_shadow/core/data/openarm_v10_arms.json từ URDF v1.0 của openarm_description.

    git clone https://github.com/enactic/openarm_description
    python scripts/extract_kinematics.py \
        openarm_description/assets/robot/openarm_v1.0/urdf/example/v1.urdf

Chỉ lấy phần động học của hai tay (gốc, trục, giới hạn khớp, điểm TCP), không lấy mesh.
File JSON đi kèm repo được sinh từ openarm_description commit 14ff67b (Apache-2.0).
"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def floats(s, default="0 0 0"):
    return [float(v) for v in (s or default).split()]


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    root = ET.parse(sys.argv[1]).getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}

    def origin(j):
        o = j.find("origin")
        return {"xyz": floats(o.get("xyz")), "rpy": floats(o.get("rpy"))}

    out = {"source": "openarm_description openarm_v1.0 urdf/example/v1.urdf", "arms": {}}
    for side in ("right", "left"):
        base = joints[f"openarm_{side}_openarm_body_link0_joint"]
        arm = {"base": origin(base), "joints": []}
        for i in range(1, 8):
            j = joints[f"openarm_{side}_joint{i}"]
            lim = j.find("limit")
            arm["joints"].append({
                "name": j.get("name"),
                **origin(j),
                "axis": floats(j.find("axis").get("xyz")),
                "lower": float(lim.get("lower")),
                "upper": float(lim.get("upper")),
            })
        arm["tcp"] = origin(joints[f"openarm_{side}_hand_tcp_joint"])
        # đầu ngón kẹp nằm cách link7 0.1025 m theo trục z của link7
        f = joints[f"openarm_{side}_finger_joint1"]
        arm["finger_tip"] = origin(f)
        out["arms"][side] = arm

    dst = Path(__file__).resolve().parents[1] / "openarm_shadow" / "core" / "data" / "openarm_v10_arms.json"
    dst.write_text(json.dumps(out, indent=1))
    print("wrote", dst)


if __name__ == "__main__":
    main()
