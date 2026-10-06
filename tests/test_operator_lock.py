"""Khoá người điều khiển: Pose thấy nhiều người, không nhảy sang người khác trong khung."""
import numpy as np

from openarm_shadow.vision.landmarks import L_SH, R_SH
from openarm_shadow.vision.perception import select_operator


def person(cx, width, vis=0.9):
    p = np.zeros((33, 3))
    p[:, 2] = vis
    p[L_SH, :2] = [cx + width / 2, 0.4]
    p[R_SH, :2] = [cx - width / 2, 0.4]
    return p


def test_first_pick_is_biggest_near_center():
    near, far = person(0.5, 0.30), person(0.8, 0.12)
    assert select_operator([far, near], None) == 1
    edge_big, center = person(0.95, 0.31), person(0.5, 0.28)
    assert select_operator([edge_big, center], None) == 1     # to gần bằng nhưng ở mép ảnh


def test_locked_operator_kept_even_if_other_is_bigger():
    op = person(0.45, 0.25)
    other = person(0.75, 0.35)                                 # người khác đi ngang, gần camera hơn
    moved = person(0.48, 0.25)
    assert select_operator([other, moved], op) == 1


def test_locked_operator_missing_returns_none():
    op = person(0.3, 0.25)
    other = person(0.8, 0.25)                                  # chỉ còn người khác ở xa vị trí cũ
    assert select_operator([other], op) is None


def test_hidden_shoulders_ignored():
    assert select_operator([person(0.5, 0.3, vis=0.1)], None) is None


class _Res:
    def __init__(self, people):
        P = lambda arr: [type("L", (), {"x": a[0], "y": a[1], "z": 0.0, "visibility": a[2]})() for a in arr]
        self.pose_landmarks = [P(p) for p in people]
        self.pose_world_landmarks = [P(p) for p in people]


def test_perception_keeps_lock_through_short_occlusion():
    from openarm_shadow.vision.perception import Perception
    perc = Perception.__new__(Perception)
    perc.pose_interval, perc.pose_hold, perc._pose_count, perc._pose_misses, perc._last_pose = 1, 0, 0, 0, None
    perc.lock_dist, perc.lock_keep, perc._lock = 1.0, 15, None
    op, other = person(0.4, 0.25), person(0.8, 0.30)
    seq = [[op], [other], [other], [op, other]]                # người điều khiển khuất 2 lần chạy Pose
    got = []
    for frame in seq:
        perc.pose = type("Pz", (), {"detect_for_video": lambda self, img, ts, f=frame: _Res(f)})()
        r = perc._run_pose(None, 0)
        got.append(None if r is None else round(float(r[0][L_SH, 0] + r[0][R_SH, 0]) / 2, 2))
    assert got == [0.4, None, None, 0.4]


def test_first_pick_prefers_full_body_over_cut_off_person():
    from openarm_shadow.vision.landmarks import L_HIP, R_HIP, NOSE
    standing = person(0.5, 0.26)
    seated = person(0.6, 0.30)
    seated[[NOSE, L_HIP, R_HIP], 2] = 0.05                     # ngồi sau bàn, hông khuất, đầu lệch khỏi khung
    assert select_operator([seated, standing], None) == 1
