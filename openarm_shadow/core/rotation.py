"""Phép toán trên ma trận quay / quaternion (slerp, khoảng cách góc)."""
from __future__ import annotations

import numpy as np


def matrix_to_quaternion(R):
    """Rotation matrix -> quaternion [w,x,y,z]."""
    R = np.asarray(R, float)
    q = np.empty(4)
    tr = float(np.trace(R))
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q[:] = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            q[:] = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s,
                    (R[0, 2] + R[2, 0]) / s]
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            q[:] = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s,
                    (R[1, 2] + R[2, 1]) / s]
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            q[:] = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                    (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    return q / np.linalg.norm(q)


def quaternion_to_matrix(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def slerp_rotation(R0, R1, fraction):
    q0, q1 = matrix_to_quaternion(R0), matrix_to_quaternion(R1)
    dot = float(q0 @ q1)
    if dot < 0:
        q1, dot = -q1, -dot
    dot = np.clip(dot, -1.0, 1.0)
    a = float(np.clip(fraction, 0.0, 1.0))
    if dot > 0.9995:
        q = q0 + a * (q1 - q0)
        return quaternion_to_matrix(q / np.linalg.norm(q))
    theta = np.arccos(dot)
    q = (np.sin((1.0 - a) * theta) * q0 + np.sin(a * theta) * q1) / np.sin(theta)
    return quaternion_to_matrix(q)


def rotation_distance(R0, R1):
    d = np.asarray(R0).T @ np.asarray(R1)
    return float(np.arccos(np.clip((np.trace(d) - 1.0) * 0.5, -1.0, 1.0)))
