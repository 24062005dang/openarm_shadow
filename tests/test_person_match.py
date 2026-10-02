"""Fusion 2 camera: camera phụ khoá đúng người camera 0 đang khoá (2 người cùng độ cao trong khung)."""
import types

import numpy as np

from openarm_shadow.multiview import MultiViewPerception
from openarm_shadow.perception import ArmObs, Frame, L_SH, R_SH
from test_multiview import human_world, two_cams

SIZE = np.array([640.0, 480.0])
OTHER = np.array([0.7, 0.0, 0.5])          # người thứ 2: lệch phải + lùi sau 0,5 m, CÙNG độ cao


def pose2d(cam, W):
    px = cam.project(W)
    return np.column_stack([px / SIZE, np.full(len(W), 0.95)])


def setup(depth=None, other_offset=OTHER):
    cams = two_cams()
    W = human_world()
    mvp = MultiViewPerception([object(), object()], cams, {}, parallel=False)
    arms = {"right": ArmObs(), "left": ArmObs()}
    arms["right"].s, arms["left"].s = W[R_SH], W[L_SH]       # chỉ dùng bề rộng vai
    mvp.view_frames = [Frame(arms, pose2d(cams[0], W), [], None, 0.0)]
    mvp._sizes = [SIZE, SIZE]
    sample = types.SimpleNamespace(bgr=np.zeros((480, 640, 3), np.uint8), depth_m=depth, intrinsics=None)
    op, other = pose2d(cams[1], W), pose2d(cams[1], W + other_offset)
    return mvp, sample, W, op, other


def depth_at(mvp, people):
    """Ảnh depth của camera 1: quanh 2 vai mỗi người = độ sâu thật của vai đó."""
    depth = np.zeros((480, 640), np.float32)
    for P, Wp in people:
        for i in (L_SH, R_SH):
            u, v = (P[i, :2] * SIZE).astype(int)
            depth[v - 5:v + 6, u - 5:u + 6] = mvp.cams[1].to_cam(Wp[i][None])[0, 2]
    return depth


def test_side_camera_picks_operator_of_front_camera():
    mvp, sample, _, op, other = setup()
    assert mvp._match_operator(1, sample, [other, op]) == 1
    assert mvp._match_operator(1, sample, [op, other]) == 0
    assert mvp.person_match[1]["mode"] == "cam0" and mvp.person_match[1]["ok"]


def test_side_camera_rejects_other_person_at_same_height():
    # Cùng độ cao: vai người kia nằm trên đường epipolar nên triangulate với người camera 0 vẫn ra sai số nhỏ.
    # [0.7, 0, 0.5]: bề rộng vai 3D ngoài 0,2-0,6 m; [0.4, 0, -0.2]: trong khoảng đó nhưng lệch vai MediaPipe camera 0.
    for off in ([0.7, 0, 0.5], [0.4, 0, -0.2]):
        mvp, sample, _, _, other = setup(other_offset=np.array(off))
        assert mvp._match_operator(1, sample, [other]) == -1
        mvp.person_cfg = {"width_tol": 99, "shoulder_width_m": (0, 99)}
        assert mvp._match_operator(1, sample, [other]) == 0     # bỏ kiểm tra bề rộng vai thì nhận nhầm


def test_depth_resolves_same_height_ambiguity():
    off = np.array([-0.5, 0, 0.4])                    # bề rộng vai triangulate vẫn hợp lý: chỉ depth phân biệt được
    mvp, sample, W, op, other = setup(other_offset=off)
    assert mvp._match_operator(1, sample, [other]) == 0
    sample.depth_m = depth_at(mvp, [(other, W + off), (op, W)])
    assert mvp._match_operator(1, sample, [other]) == -1
    assert mvp._match_operator(1, sample, [other, op]) == 1


def test_depth_disagreement_rejects():
    mvp, _, W, op, _ = setup()
    depth = np.zeros((480, 640), np.float32)
    for i in (L_SH, R_SH):
        u, v = (op[i, :2] * SIZE).astype(int)
        z = mvp.cams[1].to_cam(W[i][None])[0, 2]
        depth[v - 5:v + 6, u - 5:u + 6] = z + 0.6                 # depth đo được sau người 0,6 m
    sample = types.SimpleNamespace(bgr=np.zeros((480, 640, 3), np.uint8), depth_m=depth, intrinsics=None)
    assert mvp._match_operator(1, sample, [op]) == -1
    depth[depth > 0] -= 0.6                                       # depth khớp -> nhận
    assert mvp._match_operator(1, sample, [op]) == 0


def test_locked_by_fused_3d_shoulders():
    mvp, sample, W, _, _ = setup()
    mvp.view_frames = []                                          # không cần camera 0 khi đã có vai 3D
    mvp._operator_ref = (W, 0.0)
    moved = pose2d(mvp.cams[1], W + [0.05, 0.0, 0.03])
    other = pose2d(mvp.cams[1], W + OTHER)
    assert mvp._match_operator(1, sample, [other, moved]) == 1
    assert mvp.person_match[1]["mode"] == "3D"
    assert mvp._match_operator(1, sample, [other]) == -1


def test_no_reference_lets_camera_lock_itself():
    mvp, sample, _, op, _ = setup()
    mvp.view_frames = []
    assert mvp._match_operator(1, sample, [op]) is None


def test_perception_uses_cross_camera_match():
    from test_operator_lock import _Res, person
    from openarm_shadow.perception import Perception
    perc = Perception.__new__(Perception)
    perc.pose_interval, perc.pose_hold, perc._pose_count, perc._pose_misses, perc._last_pose = 1, 0, 0, 0, None
    perc.lock_dist, perc.lock_keep, perc._lock = 1.0, 15, None
    big, small = person(0.5, 0.30), person(0.8, 0.15)
    perc.pose = type("Pz", (), {"detect_for_video": lambda self, img, ts: _Res([big, small])})()
    perc.operator_match = lambda cands: 1                         # camera 0 khoá người nhỏ (xa camera phụ)
    assert np.isclose(perc._run_pose(None, 0)[0][L_SH, 0], small[L_SH, 0])
    perc.operator_match = lambda cands: -1                        # không ai khớp -> coi như hụt
    assert perc._run_pose(None, 0) is None
    perc.operator_match = lambda cands: None                      # chưa có tham chiếu -> tự khoá (giữ người cũ)
    assert np.isclose(perc._run_pose(None, 0)[0][L_SH, 0], small[L_SH, 0])
