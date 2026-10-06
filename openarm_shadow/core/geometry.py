"""Hình học cơ bản + các bài toán con (subproblem) dạng đóng.

Định nghĩa bài toán con theo SEW-Mimic (arXiv 2602.01632, mục III-C) và ik-geo
(Elias & Wen). Cài đặt ở đây là tự viết, không chép code của hai công trình đó.

    SP1(p1, p2, k)          : θ = argmin ‖R(k,θ)p1 − p2‖
    SP4(p, h, k, d)         : θ sao cho hᵀ R(k,θ) p = d (hoặc gần nhất)
    SP2(p1, p2, k1, k2)     : (θ1, θ2) = argmin ‖R(k1,θ1)p1 − R(k2,θ2)p2‖
"""
from __future__ import annotations

import numpy as np

EPS = 1e-9


def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > EPS else v * 0.0


def skew(k):
    x, y, z = k
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rot(k, theta):
    """Công thức Rodrigues: ma trận quay quanh trục k (đơn vị) góc theta."""
    k = unit(k)
    K = skew(k)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def rpy_to_matrix(r, p, y):
    """Quy ước URDF: R = Rz(y) Ry(p) Rx(r)."""
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def angle_between(a, b):
    a, b = unit(a), unit(b)
    return float(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b)))


def sp1(p1, p2, k):
    """Góc quay p1 quanh k để gần p2 nhất. Trả về (theta, bị suy biến?)."""
    k = unit(k)
    a = p1 - np.dot(p1, k) * k
    b = p2 - np.dot(p2, k) * k
    if np.linalg.norm(a) < 1e-9 or np.linalg.norm(b) < 1e-9:
        return 0.0, True          # p1 hoặc p2 song song trục: góc tuỳ ý
    return float(np.arctan2(np.dot(k, np.cross(a, b)), np.dot(a, b))), False


def sp4(p, h, k, d):
    """Các góc θ sao cho hᵀ R(k,θ) p = d. Nếu vô nghiệm: trả về góc gần nhất (1 nghiệm).

    Trả về (list theta, is_exact).
    """
    k = unit(k)
    p_par = np.dot(k, p) * k
    A = np.array([np.dot(h, np.cross(k, p)), np.dot(h, p - p_par)])   # hệ số của sinθ, cosθ
    b = d - np.dot(h, p_par)
    nA2 = float(A @ A)
    if nA2 < 1e-12:
        return [0.0], False          # suy biến: mọi θ như nhau
    x = A * b / nA2                  # nghiệm bình phương tối thiểu (gần gốc nhất)
    if nA2 > b * b:
        z = np.sqrt(nA2 - b * b) / nA2
        perp = np.array([A[1], -A[0]])
        xp, xm = x + z * perp, x - z * perp
        return [float(np.arctan2(xp[0], xp[1])), float(np.arctan2(xm[0], xm[1]))], True
    return [float(np.arctan2(x[0], x[1]))], False


def sp2(p1, p2, k1, k2):
    """Các cặp (θ1, θ2) để R(k1,θ1)p1 ≈ R(k2,θ2)p2 (p1, p2 được chuẩn hoá).

    Ý tưởng: vector chung c phải thoả k1·c = k1·p1 và k2·c = k2·p2
    → mỗi góc là một SP4. Ghép cặp bằng sai số thật, trả về tối đa 2 cặp tốt nhất.
    """
    p1, p2, k1, k2 = unit(p1), unit(p2), unit(k1), unit(k2)
    t1s, _ = sp4(p1, k2, k1, float(np.dot(k2, p2)))
    t2s, _ = sp4(p2, k1, k2, float(np.dot(k1, p1)))
    pairs = []
    for t1 in t1s:
        for t2 in t2s:
            err = np.linalg.norm(rot(k1, t1) @ p1 - rot(k2, t2) @ p2)
            pairs.append((err, t1, t2))
    pairs.sort(key=lambda x: x[0])
    best = pairs[0][0]
    out = [(t1, t2) for e, t1, t2 in pairs if e < best + 1e-6]
    return out[:2], best


def make_frame(k_left, k_right, k_bottom):
    """Khung từ 3 điểm: gốc = giữa trái/phải; y = sang trái; x = ra trước; z = lên.

    Giống Thuật toán 8 (MakeFrame) mô tả trong phụ lục SEW-Mimic; đúng với hệ toạ độ thuận tay phải.
    """
    p = 0.5 * (np.asarray(k_left) + np.asarray(k_right))
    uy = unit(np.asarray(k_left) - np.asarray(k_right))
    ux = unit(np.cross(uy, p - np.asarray(k_bottom)))
    uz = np.cross(ux, uy)
    return np.column_stack([ux, uy, uz]), p


def orthonormalize(R):
    u, _, vt = np.linalg.svd(R)
    Rn = u @ vt
    if np.linalg.det(Rn) < 0:
        u[:, -1] *= -1
        Rn = u @ vt
    return Rn


def seg_seg_distance(p1, q1, p2, q2):
    """Khoảng cách nhỏ nhất giữa hai đoạn thẳng [p1,q1] và [p2,q2]."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    if a <= EPS and e <= EPS:
        return float(np.linalg.norm(r))
    if a <= EPS:
        s, t = 0.0, np.clip(f / e, 0.0, 1.0)
    else:
        c = d1 @ r
        if e <= EPS:
            t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
        else:
            b = d1 @ d2
            den = a * e - b * b
            s = np.clip((b * f - c * e) / den, 0.0, 1.0) if den > EPS else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t, s = 1.0, np.clip((b - c) / a, 0.0, 1.0)
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))
