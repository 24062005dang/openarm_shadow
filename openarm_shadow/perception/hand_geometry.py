"""Hình học bàn tay từ 21 điểm 3D: đếm ngón xoè, khung lòng bàn tay (x hướng ngón, z pháp tuyến)."""
from __future__ import annotations

import numpy as np

from ..core.geometry import unit
from .types import FINGER_CHAINS, H_INDEX_MCP, H_MIDDLE_MCP, H_PINKY_MCP, H_WRIST


def open_finger_count(points):
    """Đếm bốn ngón chính đang duỗi từ hình học 3D; không dùng thumb vì biến thiên lớn."""
    p = np.asarray(points, float)
    if p.shape != (21, 3):
        return 0
    count = 0
    for mcp, pip, dip, tip in FINGER_CHAINS:
        ids = (H_WRIST, mcp, pip, dip, tip)
        if not np.all(np.isfinite(p[list(ids)])):
            continue
        chain = (np.linalg.norm(p[pip] - p[mcp]) + np.linalg.norm(p[dip] - p[pip]) +
                 np.linalg.norm(p[tip] - p[dip]))
        straight = np.linalg.norm(p[tip] - p[mcp]) / max(chain, 1e-6)
        reach = np.linalg.norm(p[tip] - p[H_WRIST]) / max(np.linalg.norm(p[mcp] - p[H_WRIST]), 1e-6)
        count += int(straight > 0.78 and reach > 1.30)
    return count


def palm_frame_from_depth(points, plane_normal=None, plane_quality=0.0, side=None):
    """Frame camera của tay: x hướng ngón, y út->trỏ, z=x×y; plane chỉ tinh chỉnh pháp tuyến."""
    p = np.asarray(points, float)
    ids = [H_WRIST, H_INDEX_MCP, H_MIDDLE_MCP, 13, H_PINKY_MCP]
    if p.shape != (21, 3) or not np.all(np.isfinite(p[ids])):
        return None, None
    center = np.mean(p[ids], axis=0)
    x = unit(0.5 * (p[H_MIDDLE_MCP] + p[13]) - p[H_WRIST])
    across = p[H_INDEX_MCP] - p[H_PINKY_MCP]
    y_land = unit(across - (across @ x) * x)
    z_land = unit(np.cross(x, y_land))
    # Hai bàn tay có chirality đối nhau. Bù dấu để z đều hướng ra khỏi lòng bàn tay.
    if side == "right":
        z_land = -z_land
    z = z_land
    if plane_normal is not None and np.linalg.norm(plane_normal) > 0.5:
        pn = unit(plane_normal)
        if pn @ z_land < 0:
            pn = -pn
        weight = float(np.clip(plane_quality, 0.0, 0.85))
        z = unit((1.0 - weight) * z_land + weight * pn)
    y = unit(np.cross(z, x))
    x = unit(np.cross(y, z))
    R = np.column_stack([x, y, z])
    return R, center
