"""Tham chiếu thân lọc theo độ nhất quán (perception.body.BodyRef), Kalman loại điểm theo sai số dự đoán, độ tin cậy
riêng từng điểm, độ dài xương. Test chuyển từ nhánh phongnv (commit 2653374) sang cấu trúc package của nhánh này."""
import types

import numpy as np

from openarm_shadow.config import load_config
from openarm_shadow.core.geometry import rot, unit
from openarm_shadow.core.rotations import rotation_distance
from openarm_shadow.filtering import PointKalman
from openarm_shadow.perception import L_HIP, L_SH, R_HIP, R_SH, BodyRef
from openarm_shadow.retarget.pipeline import ShadowPipeline
from test_pipeline import fake_frame

CFG = {"enabled": True, "learn_s": 0.3, "relearn_lost_s": 1.0, "adopt_s": 0.5}
PTS = {L_SH: np.array([0.18, -0.35, 1.3]), R_SH: np.array([-0.18, -0.35, 1.3]),
       L_HIP: np.array([0.1, 0.15, 1.3]), R_HIP: np.array([-0.1, 0.15, 1.3])}


def learned(cfg=CFG):
    ref = BodyRef(cfg)
    for k in range(12):
        ref.learn(np.eye(3), np.zeros(3), True, k / 30, points=PTS)
    assert ref.ready and set(ref.points) == set(PTS)
    return ref


def test_small_deviation_uses_measurement_and_reference_follows():
    ref = learned()
    meas = dict(PTS)
    meas[L_SH] = PTS[L_SH] + [0, -0.02, 0]                         # nhún vai 2 cm: chuyển động thật
    out = ref.gate_points(meas, 1.0)
    assert np.allclose(out[L_SH], meas[L_SH]) and ref.weights[L_SH] == 1.0
    assert 0 < np.linalg.norm(ref.points[L_SH] - PTS[L_SH]) < 0.002 # tham chiếu trôi chậm theo
    assert ref.hint == "theo do"


def test_occluded_point_uses_reference_and_blends_in_between():
    ref = learned()
    meas = dict(PTS)
    meas[R_HIP] = PTS[R_HIP] + [0, 0, -0.15]                       # tay che bụng: hông đo lệch ra trước 15 cm
    meas[L_SH] = PTS[L_SH] + [0.055, 0, 0]                         # vai lệch 5,5 cm: giữa 3 và 8 cm -> trộn
    out = ref.gate_points(meas, 1.0)
    assert np.allclose(out[R_HIP], PTS[R_HIP]) and ref.weights[R_HIP] == 0.0
    assert 0.3 < ref.weights[L_SH] < 0.7
    assert np.linalg.norm(out[L_SH] - PTS[L_SH]) < 0.055
    meas[L_SH] = np.full(3, np.nan)                                # vai không đo được: dùng tham chiếu
    out = ref.gate_points(meas, 1.1)
    assert np.allclose(out[L_SH], ref.points[L_SH]) and "uoc luong" in ref.status()


def test_rigid_shift_is_adopted_but_partial_occlusion_is_not():
    ref = learned()
    shift = np.array([0.12, 0.0, 0.0])                             # người bước sang 12 cm: cả thân dịch như khối cứng
    moved = {i: X + shift for i, X in PTS.items()}
    out = ref.gate_points(moved, 1.0)
    assert np.allclose(out[L_SH], PTS[L_SH])                       # chưa đủ adopt_s: vẫn tham chiếu
    for k in range(20):
        out = ref.gate_points(moved, 1.0 + k / 30)
    assert np.allclose(out[L_SH], moved[L_SH]) and np.allclose(ref.points[R_HIP], moved[R_HIP])
    ref = learned()
    bent = dict(PTS)
    for i in (L_HIP, R_HIP):                                       # chỉ 2 hông lệch (tay che): không phải khối cứng
        bent[i] = PTS[i] + [0, 0, -0.12]
    for k in range(60):
        ref.gate_points(bent, 1.0 + k / 30)
    assert np.allclose(ref.points[L_HIP], PTS[L_HIP])              # che lâu vẫn không nhận vị trí sai


def test_rotation_gating_single_camera():
    ref = BodyRef(CFG)
    for k in range(12):
        ref.learn(np.eye(3), None, True, k / 30)
    small, big = rot(np.array([0.0, 1.0, 0.0]), np.deg2rad(3)), rot(np.array([0.0, 1.0, 0.0]), np.deg2rad(30))
    assert rotation_distance(ref.gate_rotation(small, 1.0), small) < 1e-9          # lệch nhỏ: theo đo
    assert rotation_distance(ref.gate_rotation(big, 1.1), np.eye(3)) < np.deg2rad(0.2)   # lệch lớn: tham chiếu
    for k in range(40):                                            # lệch lớn liên tục 2 x adopt_s: xoay thật
        R = ref.gate_rotation(big, 1.2 + k / 30)
    assert rotation_distance(R, big) < 1e-9


def test_relearn_after_lost_and_reset():
    ref = learned()
    ref.lost(1.0)
    ref.lost(1.5)
    assert ref.ready
    ref.lost(2.1)
    assert not ref.ready
    ref = learned()
    ref.reset()
    assert not ref.ready and BodyRef({"enabled": False}).gate_rotation(np.eye(3), 0.0) is not None


def test_perception_body_frame_ignores_occluded_hips():
    from test_depth import _fake_perception, _scene

    def run(cfg):
        p = _fake_perception()
        p.body_ref = BodyRef(cfg)
        pres, _, hres = _scene()
        p.pose = types.SimpleNamespace(detect_for_video=lambda img, ts: pres)
        p.hands = types.SimpleNamespace(detect_for_video=lambda img, ts: hres)
        img = np.zeros((480, 640, 3), np.uint8)
        for k in range(15):
            fr0 = p.process(img, t=k / 30)
        world = pres.pose_world_landmarks[0]
        for i in (L_HIP, R_HIP):                                   # tay trước bụng: hông đo lệch ra trước 15 cm
            world[i] = types.SimpleNamespace(x=world[i].x, y=world[i].y, z=-0.15, visibility=world[i].visibility)
        for k in range(15, 50):
            fr = p.process(img, t=k / 30)
        return np.rad2deg(rotation_distance(fr0.body_R, fr.body_R))

    drift_off = run({"enabled": False})
    drift_on = run({**CFG, "adopt_s": 5.0})
    assert drift_off > 10 and drift_on < 1.0


def _fusion(ref_cfg):
    from test_multiview import FakeView, hand_points, human_world, two_cams
    from openarm_shadow.cameras import MultiSample
    from openarm_shadow.fusion import MultiViewPerception

    class View(FakeView):
        hide = ()

        def process(self, bgr, t=None):
            fr = super().process(bgr, t)
            for i in self.hide:
                fr.pose_2d[i, 2] = 0.0                                # điểm bị tay che hẳn
            return fr

    cams, W = two_cams(), human_world()
    R_hand = np.column_stack([unit([0.2, -0.8, -0.4]), unit([1, 0.3, 0.2]), [0, 0, 0]])
    R_hand[:, 1] = unit(R_hand[:, 1] - (R_hand[:, 1] @ R_hand[:, 0]) * R_hand[:, 0])
    R_hand[:, 2] = np.cross(R_hand[:, 0], R_hand[:, 1])
    views = [View(c, W, hand_points("right", R_hand, W[16]), seed=i, noise=0.3) for i, c in enumerate(cams)]
    mvp = MultiViewPerception(views, cams, {"reproj_thresh_px": 25}, parallel=False, body_ref_cfg=ref_cfg)
    blank = types.SimpleNamespace(bgr=np.zeros((480, 640, 3), np.uint8), depth_m=None, intrinsics=None)
    return mvp, views, W, (lambda k: mvp.process(MultiSample([blank, blank], k / 30))), blank


def test_fusion_shoulder_occlusion_vs_real_shrug():
    from openarm_shadow.perception import body_frame

    def upper_arm_error(cfg, dz):
        mvp, views, W, step, _ = _fusion(cfg)
        for k in range(15):
            step(k)
        Rb, _ = body_frame(W, np.ones(33))
        W0 = W.copy()
        W[12] = W[12] + [0.0, 0.0, dz]                             # vai phải đo lệch ra trước dz
        for k in range(15, 25):
            fr = step(k)
        ob = fr.arms["right"]
        u_true = unit(Rb.T @ (W0[14] - W0[12]))
        return np.degrees(np.arccos(np.clip(unit(ob.e - ob.s) @ u_true, -1, 1))), mvp

    err_off, _ = upper_arm_error({"enabled": False}, -0.10)
    err_on, mvp = upper_arm_error({**CFG, "adopt_s": 5.0}, -0.10)
    assert mvp.body_ref.ready and R_SH in mvp.body_ref.points
    assert err_off > 10 and err_on < 3                             # tay che, vai đo lệch 10 cm: dùng tham chiếu
    _, mvp = upper_arm_error({**CFG, "adopt_s": 5.0}, -0.015)      # vai lệch 1,5 cm (cử động thật nhỏ): theo đo
    assert mvp.body_ref.weights[R_SH] == 1.0


def test_fusion_keeps_arm_when_shoulders_hidden_and_draws_torso():
    mvp, views, W, step, blank = _fusion(CFG)
    for k in range(15):
        step(k)
    for v in views:
        v.hide = (L_SH, R_SH)
    fr = step(16)
    assert fr.body_R is not None and fr.arms["right"].s is not None
    from openarm_shadow.cameras import MultiSample
    assert mvp.draw(MultiSample([blank, blank], 16 / 30), fr).shape[0] == 360
    for k in range(17, 60):                                       # che lâu hơn relearn_lost_s: học lại
        step(k)
    assert not mvp.body_ref.ready


def test_kalman_innovation_gating():
    kf = PointKalman(q=6.0, r=0.015, gate_sigma=4.0, gate_min_m=0.05, confirm_frames=3)
    for k in range(30):
        kf(np.zeros((2, 3)), k / 30)
    z = np.zeros((2, 3))
    z[1, 0] = 0.15                                                # điểm 2 vọt 15 cm (bị che, đặt nhầm)
    x = kf(z, 1.0)
    assert list(kf.rejected) == [False, True] and abs(x[1, 0]) < 0.01
    x = kf(z, 1.0 + 1 / 30)
    assert kf.rejected[1]
    x = kf(z, 1.0 + 2 / 30)                                       # lệch liên tục: chuyển động thật -> nhận lại
    assert not kf.rejected[1] and x[1, 0] > 0.1                   # (bất định dự đoán tăng: lọt ngưỡng hoặc confirm)
    kf2 = PointKalman(q=6.0, r=0.015)                             # gate_sigma = 0: như cũ, không loại
    for k in range(30):
        kf2(np.zeros((2, 3)), k / 30)
    kf2(z, 1.0)
    assert not kf2.rejected.any()


def test_bone_length_fix_for_rejected_elbow():
    cfg = load_config()
    cfg["mapping"]["robot_arms"] = ["right"]
    cfg["filter"]["arm_shape"] = {"enabled": True, "min_samples": 5}
    pipe = ShadowPipeline(cfg)
    s, e, w = np.zeros(3), np.array([0, 0, -0.30]), np.array([0.26, 0, -0.30])
    for k in range(10):
        pipe.arm_shape["right"](s, e, w, k / 30)
    pipe.kf_rejected["right"] = np.array([False, True, True])
    out = pipe._fix_bone_length("right", np.stack([s, np.array([0, 0, -0.20]), np.array([0.40, 0, -0.20])]))
    assert np.isclose(np.linalg.norm(out[1] - out[0]), 0.30) and np.isclose(np.linalg.norm(out[2] - out[1]), 0.26)


def test_kalman_per_point_confidence():
    rng = np.random.default_rng(2)
    meas = [rng.normal(0, 0.01, (3, 3)) for _ in range(40)]
    a, b = PointKalman(), PointKalman()
    for k, m in enumerate(meas):                         # mảng độ tin cậy bằng nhau = một số chung (như trước)
        assert np.allclose(a(m, k / 30, 0.7), b(m, k / 30, [0.7, 0.7, 0.7]))
    kf = PointKalman(q=6.0, r=0.015)
    for k in range(30):
        kf(np.zeros((3, 3)), k / 30)
    jump = np.zeros((3, 3))
    jump[:, 0] = 0.2                                     # cả 3 điểm vọt 20 cm, chỉ cổ tay nhìn kém
    x = kf(jump, 1.0, conf=[1.0, 1.0, 0.05])
    assert x[0, 0] > 0.1 and x[1, 0] > 0.1               # vai, khuỷu nhìn rõ: bám đo
    assert x[2, 0] < 0.02                                # cổ tay nhìn kém: tin dự đoán


def test_pipeline_kalman_uses_per_point_confidence():
    cfg = load_config()
    cfg["mapping"]["robot_arms"] = ["right"]
    cfg["filter"]["landmark_kalman"] = {"enabled": True, "per_point_conf": True}
    q = {"right": np.deg2rad([20, 20, 0, 60, 0, 0, 0])}

    def wrist_after_jump(points_conf):
        pipe = ShadowPipeline(cfg)
        pipe.seed({"right": np.zeros(8)})
        for i in range(30):
            fr = fake_frame(pipe, q, i / 15)
            fr.arms["right"].conf["points"] = (0.95, 0.95, 0.95)
            pipe.step(fr)
        fr = fake_frame(pipe, q, 2.0)
        ob = fr.arms["right"]
        w0 = ob.w.copy()
        ob.w = ob.w + [0.1, 0.0, 0.0]                    # cổ tay đo vọt 10 cm
        ob.conf["points"] = points_conf
        pipe.step(fr)
        return np.linalg.norm(pipe.lm_kf["right"].x[2] - w0)

    assert wrist_after_jump((0.95, 0.95, 0.95)) > 0.05  # cổ tay nhìn rõ: Kalman nhận bước nhảy
    assert wrist_after_jump((0.95, 0.95, 0.05)) < 0.01  # vai/khuỷu rõ nhưng cổ tay kém: không tin điểm cổ tay
