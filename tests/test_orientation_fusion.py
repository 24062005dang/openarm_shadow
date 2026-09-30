"""Hướng bàn tay nhiều nguồn (OrientationFusion) và chế độ fusion nhẹ body_source=front."""
import types

import numpy as np

from openarm_shadow.geometry import rot, unit
from openarm_shadow.handfusion import OrientationFusion
from openarm_shadow.multiview import MultiSample, MultiViewPerception, view_options
from openarm_shadow.perception import ArmObs, Frame, body_frame, palm_frame_from_depth

from test_handfusion import _hand_R, ang
from test_multiview import FakeView, hand_points, human_world, two_cams

FLIP = np.diag([1.0, -1.0, -1.0])


def test_flip_from_detector_is_rejected_when_other_sources_agree():
    of = OrientationFusion()
    R0 = rot([0, 0, 1], 0.3)
    for _ in range(3):
        of.update(R0, 0.9, [(R0, 0.5, "rgb")], base_strong=True)
    R, state, conf, used = of.update(R0 @ FLIP, 0.9, [(R0, 0.5, "rgb")], base_strong=True)
    assert ang(R, R0) < 2 and conf > 0.6                  # chọn giả thuyết lật lại -> vẫn đúng hướng cũ


def test_big_real_rotation_needs_confirmation_and_freezes_wrist_meanwhile():
    of = OrientationFusion(switch_deg=100, switch_frames=3)
    # quay 150° quanh pháp tuyến lòng bàn tay (không phải quanh trục ngón): cả hai giả thuyết đều xa hướng cũ
    R0, R1 = np.eye(3), rot([0, 0, 1], np.deg2rad(150))
    of.update(R0, 0.9, base_strong=True)
    confs = []
    for _ in range(2):
        R, state, conf, _ = of.update(R1, 0.9, base_strong=True)
        confs.append(conf)
        assert ang(R, R0) < 1                            # chưa chắc: giữ hướng cũ
    assert confs == [0.0, 0.0]                           # và đóng băng J5-J7
    R, state, conf, _ = of.update(R1, 0.9, base_strong=True)
    assert conf > 0.6 and ang(R, R1) < 20                # khung thứ 3: nhận


def test_fast_rotation_is_tracked_with_little_lag():
    of = OrientationFusion()
    lag = []
    for k in range(30):                                  # xoay cổ tay 8°/khung (~240°/s ở 30 fps)
        Rt = rot([1, 0, 0], np.deg2rad(8 * k))
        R, *_ = of.update(Rt, 0.9, base_strong=True)
        lag.append(ang(R, Rt))
    assert max(lag[5:]) < 6, max(lag[5:])


def test_missing_holds_with_zero_conf_then_lost():
    of = OrientationFusion(hold_frames=2)
    of.update(np.eye(3), 0.9, base_strong=True)
    for _ in range(2):
        R, state, conf, _ = of.update(None)
        assert state == "HOLD" and conf == 0.0 and R is not None
    R, state, conf, _ = of.update(None)
    assert state == "LOST" and R is None


def test_single_extra_source_is_degraded_but_usable():
    of = OrientationFusion()
    R, state, conf, used = of.update(None, extras=[(np.eye(3), 0.5, "rgb:front")])
    assert state == "DEGRADED" and conf == 0.7 and used == ["rgb:front"]


def test_view_options_front_mode():
    cfg = {"fusion": {"body_source": "front", "cameras": [{"name": "front"}, {"name": "side45"}]},
           "models": {"pose_interval": 2, "pose_hold_frames": 6}, "mapping": {"robot_arms": ["right"], "mode": "direct"}}
    o = view_options(cfg)
    assert [x["pose_enabled"] for x in o] == [True, False]
    assert all(x["force_hand_side"] == "right" and x["pose_interval"] == 2 for x in o)
    assert [x["num_hands"] for x in o] == [2, 1]          # camera có Pose tìm 2 tay để gán theo cổ tay
    cfg["mapping"]["mode"] = "mirror"
    assert view_options(cfg)[0]["force_hand_side"] == "left"
    cfg["fusion"]["body_source"] = "triangulate"
    cfg["mapping"]["robot_arms"] = ["right", "left"]
    o = view_options(cfg)
    assert all(x["pose_enabled"] and x["force_hand_side"] is None and x["num_hands"] == 2 for x in o)


class FrontView(FakeView):
    """Camera 0 như Perception thật: tay + vai/khuỷu/cổ tay trong khung thân (từ điểm world)."""

    def process(self, bgr, t=None):
        fr = super().process(bgr, t)
        Rb, origin = body_frame(self.world, np.ones(33))
        ob = fr.arms["right"]
        ob.s, ob.e, ob.w = (Rb.T @ (self.world[k] - origin) for k in (12, 14, 16))
        ob.conf["upper"] = ob.conf["fore"] = 0.9
        fr.body_R, fr.body_origin = Rb, origin
        return fr


class HandOnlyView(FakeView):
    def process(self, bgr, t=None):
        fr = super().process(bgr, t)
        fr.pose_2d = None
        return fr


def test_front_mode_end_to_end():
    cams = two_cams()
    W = human_world()
    R_hand = _hand_R()
    hand = hand_points("right", R_hand, W[16])
    views = [FrontView(cams[0], W, hand, seed=0), HandOnlyView(cams[1], W, hand, seed=1)]
    mvp = MultiViewPerception(views, cams, {"reproj_thresh_px": 25, "body_source": "front"}, parallel=False)
    blank = types.SimpleNamespace(bgr=np.zeros((480, 640, 3), np.uint8), depth_m=None, intrinsics=None)
    for k in range(20):
        fr = mvp.process(MultiSample([blank, blank], 0.033 * k))
    ob = fr.arms["right"]
    Rb, _ = body_frame(W, np.ones(33))
    assert fr.fusion["body"] == "front" and ob.conf["upper"] == 0.9
    assert np.allclose(ob.w - ob.e, Rb.T @ (W[16] - W[14]))
    R_true, _ = palm_frame_from_depth(hand, side="right")
    assert ang(Rb @ ob.H, R_true) < 8
    assert ob.conf["hand"] >= 0.6 and fr.fusion["hand_right"]["views"] == 2


def test_no_flip_lock_after_hand_returns():
    """Mất tay rồi quay lại ở hướng lật 150° quanh trục ngón, các nguồn phụ đều đồng ý hướng thật:
    không được khoá nhầm giả thuyết lật với độ tin cậy cao."""
    of = OrientationFusion()
    of.update(np.eye(3), 0.9, base_strong=True)
    for _ in range(3):
        of.update(None)                                    # HOLD
    Rt = rot([1, 0, 0], np.deg2rad(150))
    extras = [(Rt, 0.5, "rgb:front"), (Rt, 0.6, "depth:side45")]
    for _ in range(6):
        R, state, conf, _ = of.update(Rt @ FLIP, 0.9, extras, base_strong=True)   # 3D đưa ra giả thuyết lật
        if conf > 0:
            assert ang(R, Rt) < 30, (state, ang(R, Rt))      # khi đã điều khiển thì phải là hướng thật
    assert conf > 0 and ang(R, Rt) < 20


def test_side_camera_seeing_other_hand_is_rejected_in_front_mode():
    cams = two_cams()
    W = human_world()
    hand = hand_points("right", _hand_R(), W[16])
    wrong = hand - W[16] + W[15]                          # D435i thấy bàn tay TRÁI (chỉ chạy Hand, tự gán "right")
    views = [FrontView(cams[0], W, hand, seed=0), HandOnlyView(cams[1], W, wrong, seed=1)]
    mvp = MultiViewPerception(views, cams, {"reproj_thresh_px": 25, "body_source": "front"}, parallel=False)
    blank = types.SimpleNamespace(bgr=np.zeros((480, 640, 3), np.uint8), depth_m=None, intrinsics=None)
    fr = mvp.process(MultiSample([blank, blank], 0.0))
    assert fr.fusion["hand_right"]["rejected"] == 1 and fr.fusion["hand_right"]["views"] == 1
