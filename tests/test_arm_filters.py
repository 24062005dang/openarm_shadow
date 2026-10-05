"""Lọc khung xương cánh tay (ArmShape) và Kalman điểm 3D (PointKalman)."""
import numpy as np

from openarm_shadow.config import load_config
from openarm_shadow.filtering import ArmShape, PointKalman
from openarm_shadow.pipeline import ShadowPipeline
from test_pipeline import fake_frame


def arm(up=0.30, fo=0.26, rng=None):
    s = np.zeros(3)
    e = s + [0.0, 0.0, -up]
    w = e + [fo, 0.0, 0.0]
    if rng is not None:
        e, w = e + rng.normal(0, 0.005, 3), w + rng.normal(0, 0.005, 3)
    return s, e, w


def test_arm_shape_learns_then_rejects_wrong_elbow():
    a, rng = ArmShape(min_samples=10), np.random.default_rng(0)
    for k in range(20):
        ok_u, ok_f, info = a(*arm(rng=rng), k / 30)
        assert ok_u and ok_f
    assert abs(info["ref"][0] - 0.30) < 0.01
    s, e, w = arm()
    e_bad = e + [-0.2, 0.0, 0.0]                         # khuỷu "nhảy" ra sau thân 20 cm (depth rơi vào thân)
    ok_u, ok_f, info = a(s, e_bad, w, 21 / 30)
    # Lệch gần vuông góc cánh tay trên: đoạn trên chỉ dài thêm 20% (lọt), cẳng tay dài 1.8 lần -> bị bắt
    assert ok_u and not ok_f
    assert abs(a.ref()[0] - 0.30) < 0.01                 # khung sai không vào thống kê
    assert a(*arm(), 22 / 30)[:2] == (True, True)


def test_arm_shape_relearns_after_person_lost():
    a = ArmShape(min_samples=5, reset_s=2.0)
    for k in range(10):
        a(*arm(up=0.30), k / 30)
    assert not a(*arm(up=0.40), 0.5)[0]                  # người khác tay dài hơn 33% -> bị coi là sai
    for k in range(10):                                  # mất người > 2 s rồi người mới vào: học lại
        a(*arm(up=0.40), 3.0 + k / 30)
    assert a(*arm(up=0.40), 3.5)[0] and abs(a.ref()[0] - 0.40) < 1e-6


def test_pipeline_holds_upper_joints_on_bad_segment():
    cfg = load_config()
    cfg["mapping"]["robot_arms"] = ["right"]
    cfg["filter"]["arm_shape"] = {"enabled": True, "min_samples": 10}
    pipe = ShadowPipeline(cfg)
    pipe.seed({"right": np.zeros(8)})
    q = {"right": np.deg2rad([20, 20, 0, 60, 0, 0, 0])}
    for i in range(30):
        pipe.step(fake_frame(pipe, q, i / 15))
    assert not pipe.held["right"][0]
    fr = fake_frame(pipe, q, 2.0)
    ob = fr.arms["right"]
    ob.e = ob.s + 2.0 * (ob.e - ob.s)                    # cánh tay trên dài gấp đôi: khuỷu sai
    pipe.step(fr)
    assert pipe.held["right"][:4].all() and pipe.arm_shape_info["right"]["ok"][0] is False
    fr = fake_frame(pipe, q, 2.1)
    fr.arms["right"].w = fr.arms["right"].w + [0.3, 0, 0]  # chỉ cẳng tay sai -> vẫn giữ cả J1-J4
    pipe.step(fr)
    assert pipe.held["right"][:4].all() and pipe.arm_shape_info["right"]["ok"] == [True, False]


def test_kalman_smooths_noise_and_follows_motion():
    rng = np.random.default_rng(1)
    kf, t = PointKalman(q=6.0, r=0.015), np.arange(0, 3, 1 / 30)
    truth = np.stack([0.2 * np.sin(2 * np.pi * 0.5 * t), np.zeros_like(t), np.zeros_like(t)], 1)[:, None, :]
    meas = truth + rng.normal(0, 0.01, truth.shape)
    out = np.array([kf(m, ti) for m, ti in zip(meas, t)])
    err_raw = np.sqrt(np.mean((meas[30:] - truth[30:]) ** 2))
    err_kf = np.sqrt(np.mean((out[30:] - truth[30:]) ** 2))
    assert err_kf < 0.8 * err_raw                        # bớt nhiễu
    assert np.max(np.abs(out[30:] - truth[30:])) < 0.03  # vẫn bám theo chuyển động 0,5 Hz biên độ 20 cm


def test_kalman_low_confidence_trusts_prediction_and_resets_after_gap():
    kf = PointKalman(q=6.0, r=0.015, max_gap_s=0.3)
    for k in range(30):
        kf(np.zeros((1, 3)), k / 30)
    x = kf(np.array([[0.2, 0, 0]]), 1.0, conf=0.05)      # điểm vọt 20 cm nhưng tin cậy rất thấp
    assert x[0, 0] < 0.02
    x = kf(np.array([[0.2, 0, 0]]), 2.0)                 # mất điểm > max_gap_s: nhận điểm mới ngay
    assert np.isclose(x[0, 0], 0.2)
