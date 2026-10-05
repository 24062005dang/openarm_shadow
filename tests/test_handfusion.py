"""Làm sạch bàn tay 3D: gán tay theo cổ tay, lọc đốt bất thường, Kabsch lòng bàn tay, chặn theo cẳng tay,
bỏ bàn tay sai camera, cổ tay chung cho cẳng tay và bàn tay."""
import types

import numpy as np

from openarm_shadow.core.geometry import rot, unit
from openarm_shadow.handfusion import HandShape, PalmModel, hand_forearm_angle
from openarm_shadow.perception import assign_hands_to_wrists
from openarm_shadow.cameras import MultiSample
from openarm_shadow.multiview import HandOrientationTracker, MultiViewPerception
from openarm_shadow.perception import palm_frame_from_depth

from test_multiview import FakeView, SUBJECT, hand_points, human_world, two_cams


def rand_R(rng):
    q = unit(rng.normal(size=4))
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def ang(R1, R2):
    return np.degrees(np.arccos(np.clip((np.trace(R1.T @ R2) - 1) / 2, -1, 1)))


def test_assign_hands_unique_and_gated():
    wrists = {"right": (100, 200), "left": (300, 200)}
    # hai bàn tay đều gần cổ tay phải hơn: bàn tay gần nhất lấy "right", bàn tay kia lấy "left" nếu trong ngưỡng
    assert assign_hands_to_wrists([(110, 200), (200, 200)], wrists, 150) == ["right", "left"]
    assert assign_hands_to_wrists([(200, 200), (110, 200)], wrists, 150) == ["left", "right"]
    # bàn tay xa mọi cổ tay (người khác phía sau) bị bỏ
    assert assign_hands_to_wrists([(600, 50)], wrists, 150) == [None]
    assert assign_hands_to_wrists([(110, 200)], {"right": None, "left": (300, 200)}, 150) == [None]


def test_hand_shape_drops_implausible_bone():
    rng = np.random.default_rng(0)
    shape = HandShape()
    for _ in range(20):
        P = hand_points("right", rand_R(rng), SUBJECT) + rng.normal(0, 0.002, (21, 3))
        shape.filter(P)
    P = hand_points("right", np.eye(3), SUBJECT)
    bad = P.copy()
    bad[8] += [0.08, 0.0, 0.0]                      # đầu ngón trỏ bay xa 8 cm (landmark nhiễu)
    out, dropped = shape.filter(bad)
    assert dropped >= 1 and not np.all(np.isfinite(out[8]))
    assert np.allclose(out[:8], P[:8])              # các điểm còn lại giữ nguyên, không bịa điểm


def test_palm_kabsch_robust_to_one_bad_knuckle():
    rng = np.random.default_rng(1)
    pm = PalmModel("right")
    for _ in range(pm.learn_frames):
        R = rand_R(rng)
        pm.estimate(hand_points("right", R, SUBJECT) + rng.normal(0, 0.002, (21, 3)), quality=1.0)
    assert pm.template is not None
    e3, ek = [], []
    for _ in range(60):
        R = rand_R(rng)
        P = hand_points("right", R, SUBJECT)
        R_true, _ = palm_frame_from_depth(P, side="right")
        Pn = P + rng.normal(0, 0.003, (21, 3))
        Pn[9] += unit(rng.normal(size=3)) * 0.03        # gốc ngón giữa lệch 3 cm (bị che)
        R3, _ = palm_frame_from_depth(Pn, side="right")
        Rk, _, rms, mode = pm.estimate(Pn)
        assert mode == "KABSCH" and Rk is not None
        e3.append(ang(R3, R_true))
        ek.append(ang(Rk, R_true))
    # nhiễu 3 mm mỗi điểm + 1 gốc ngón lệch 3 cm: Kabsch 5 điểm bỏ được điểm lệch, cách 3 điểm thì không
    assert np.median(ek) < 5.0, np.median(ek)
    assert np.median(e3) > 1.8 * np.median(ek), (np.median(e3), np.median(ek))


def test_palm_kabsch_missing_knuckle_still_works():
    rng = np.random.default_rng(2)
    pm = PalmModel("left")
    for _ in range(pm.learn_frames):
        pm.estimate(hand_points("left", rand_R(rng), SUBJECT), quality=1.0)
    R = rand_R(rng)
    P = hand_points("left", R, SUBJECT)
    R_true, _ = palm_frame_from_depth(P, side="left")
    P[17] = np.nan
    Rk, _, _, mode = pm.estimate(P)
    assert mode == "KABSCH" and ang(Rk, R_true) < 1.0


def test_hand_forearm_angle():
    R = np.eye(3)                                   # ngón tay theo +x
    assert hand_forearm_angle(R, [0, 0, 0], [1, 0, 0]) < 1
    assert abs(hand_forearm_angle(R, [0, 0, 0], [0, 1, 0]) - 90) < 1
    assert hand_forearm_angle(R, [0, 0, 0], [-1, 0, 0]) > 170       # bàn tay chỉ ngược cẳng tay: sai


def test_tracker_less_lag_when_view_is_clear():
    R0, R1 = np.eye(3), rot([0, 0, 1], np.deg2rad(40))
    slow, fast = HandOrientationTracker(alpha=0.5, alpha_max=0.9), HandOrientationTracker(alpha=0.5, alpha_max=0.9)
    slow.update(R0, 1.0), fast.update(R0, 1.0)
    Rs, _ = slow.update(R1, 0.36)                   # nhìn kém: làm mượt nhiều
    Rf, _ = fast.update(R1, 1.0)                    # nhìn rõ: bám nhanh
    assert ang(Rf, R1) < ang(Rs, R1)
    assert ang(Rf, R1) < 5


def _hand_R():
    R = np.column_stack([unit([0.2, -0.8, -0.4]), unit([1, 0.3, 0.2]), [0, 0, 0]])
    R[:, 1] = unit(R[:, 1] - (R[:, 1] @ R[:, 0]) * R[:, 0])
    R[:, 2] = np.cross(R[:, 0], R[:, 1])
    return R


def _run(views, cams, frames=3):
    mvp = MultiViewPerception(views, cams, {"reproj_thresh_px": 25}, parallel=False)
    blank = types.SimpleNamespace(bgr=np.zeros((480, 640, 3), np.uint8), depth_m=None, intrinsics=None)
    fr = None
    for k in range(frames):
        fr = mvp.process(MultiSample([blank, blank], 0.033 * k))
    return fr


def test_wrong_hand_in_one_view_is_rejected():
    """Camera 1 gán nhầm bàn tay đang ở cổ tay TRÁI thành tay phải: bị loại, không trộn vào tay phải."""
    cams = two_cams()
    W = human_world()
    hand = hand_points("right", _hand_R(), W[16])
    wrong = hand - W[16] + W[15]
    views = [FakeView(cams[0], W, hand, seed=0), FakeView(cams[1], W, wrong, seed=1)]
    fr = _run(views, cams)
    assert fr.fusion["hand_right"]["rejected"] == 1
    assert fr.fusion["hand_right"]["views"] == 1


def test_forearm_and_hand_share_wrist():
    cams = two_cams()
    W = human_world()
    hand = hand_points("right", _hand_R(), W[16] + [0.01, 0.0, 0.0])   # cổ tay Hand lệch cổ tay Pose 1 cm
    views = [FakeView(c, W, hand, seed=i, noise=0.3) for i, c in enumerate(cams)]
    fr = _run(views, cams, frames=10)               # tỉ lệ trộn tăng dần 0,15/khung tới 0,7
    assert fr.fusion["points"][16].get("hand_root")
    ob = fr.arms["right"]
    Rb = fr.body_R
    w_world = Rb @ ob.w + fr.body_origin
    # cổ tay dùng cho cẳng tay dịch về phía cổ tay của bàn tay (70%)
    assert abs(np.linalg.norm(w_world - W[16]) - 0.007) < 0.003


def test_end_to_end_switches_to_kabsch_and_orientation_is_right():
    from openarm_shadow.perception import body_frame
    cams = two_cams()
    W = human_world()
    R_hand = _hand_R()
    hand = hand_points("right", R_hand, W[16])
    views = [FakeView(c, W, hand, seed=i, noise=1.0) for i, c in enumerate(cams)]
    fr = _run(views, cams, frames=25)
    info = fr.fusion["hand_right"]
    assert info["fit"] == "KABSCH" and info["fit_mm"] < 6
    R_true, _ = palm_frame_from_depth(hand, side="right")
    Rb, _ = body_frame(W, np.ones(33))
    assert ang(Rb @ fr.arms["right"].H, R_true) < 8


def test_camera_weight_changes_mixing_not_confidence():
    """weight 0.7 cho webcam chỉ đổi cách trộn; độ tin cậy vai/khuỷu/bàn tay không bị kéo xuống dưới min_conf."""
    cams = two_cams()
    W = human_world()
    hand = hand_points("right", _hand_R(), W[16])
    views = [FakeView(c, W, hand, seed=i) for i, c in enumerate(cams)]
    cfg = {"reproj_thresh_px": 25, "cameras": [{"name": "front", "weight": 0.7}, {"name": "side45"}]}
    mvp = MultiViewPerception(views, cams, cfg, parallel=False)
    blank = types.SimpleNamespace(bgr=np.zeros((480, 640, 3), np.uint8), depth_m=None, intrinsics=None)
    for k in range(6):                                  # qua giai đoạn ACQUIRE của hướng tay
        fr = mvp.process(MultiSample([blank, blank], 0.033 * k))
    ob = fr.arms["right"]
    assert abs(ob.conf["upper"] - 0.95) < 1e-6 and abs(ob.conf["fore"] - 0.95) < 1e-6
    assert ob.conf["hand"] >= 0.85
