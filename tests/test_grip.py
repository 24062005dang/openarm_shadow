"""Kẹp: mở liên tục theo hai ngón, tuỳ chọn theo mức (có trễ + giữ), hiệu chuẩn theo người, mất tay thì giữ."""
import numpy as np

from openarm_shadow.config import load_config
from openarm_shadow.retarget.grip import GripMapper
from openarm_shadow.retarget.pipeline import ShadowPipeline
from test_pipeline import fake_frame


def test_continuous_mapping():
    g = GripMapper(0.2, 1.0)
    assert g(0.2, 0) == 0.0 and g(1.0, 0.1) == 1.0 and abs(g(0.6, 0.2) - 0.5) < 1e-9
    assert g(0.05, 0.3) == 0.0 and g(1.4, 0.4) == 1.0
    assert np.isnan(g(None, 0.5))


def test_levels_with_hysteresis_and_dwell():
    g = GripMapper(0.0, 1.0, levels=[0, 0.5, 1], hysteresis=0.05, dwell_s=0.15)
    assert g(0.0, 0.0) == 0.0
    t = 0.0
    for r in (0.27, 0.23, 0.28, 0.22):            # dao động quanh ranh giới 0.25: không đổi mức
        t += 0.05
        assert g(r, t) == 0.0
    assert g(0.45, 0.30) == 0.0                     # vượt ranh giới nhưng chưa giữ đủ 0.15 s
    assert g(0.45, 0.40) == 0.0
    assert g(0.45, 0.46) == 0.5                     # giữ đủ -> mở vừa
    assert g(0.95, 0.50) == 0.5 and g(0.95, 0.70) == 1.0
    assert g(0.05, 0.75) == 1.0 and g(0.05, 0.95) == 0.0   # chụm nhanh: nhảy thẳng về đóng


def test_calibration_sets_range_from_user():
    g = GripMapper(0.25, 0.9, calib_s=2.0)
    g.start_calibration(0.0)
    t = 0.0
    for k in range(60):                             # người này chụm tới 0.15, xoè tới 1.3
        t = k / 30
        g(0.15 + 1.15 * (0.5 + 0.5 * np.sin(k / 3)), t)
    g(0.7, 2.1)
    pinch, open_ = g.calib_result
    assert 0.15 < pinch < 0.35 and 1.1 < open_ < 1.3
    assert g(0.15, 2.2) == 0.0 and g(1.3, 2.3) == 1.0


def test_calibration_rejects_small_range():
    g = GripMapper(0.25, 0.9, calib_s=1.0)
    g.start_calibration(0.0)
    for k in range(40):
        g(0.5 + 0.01 * np.sin(k), k / 30)
    assert isinstance(g.calib_result, str) and g.pinch == 0.25


def test_pipeline_grip_holds_when_hand_lost():
    cfg = load_config()
    cfg["mapping"]["robot_arms"] = ["right"]
    pipe = ShadowPipeline(cfg)
    pipe.seed({"right": np.zeros(8)})
    q = {"right": np.deg2rad([20, 20, 0, 60, 0, 0, 0])}
    for i in range(30):
        out = pipe.step(fake_frame(pipe, q, i / 15))
    g0 = out["right"][7]
    assert 0 <= g0 <= 1
    fr = fake_frame(pipe, q, 2.1)
    fr.arms["right"].grip = None                    # mất ngón cái/trỏ
    out = pipe.step(fr)
    assert out["right"][7] == g0 and pipe.held["right"][7]


def test_grip_follows_fingers_when_palm_orientation_uncertain():
    # Fusion 2 camera: chụm tay làm hướng lòng bàn tay về HOLD (conf hand thấp) nhưng 2 ngón vẫn thấy rõ ->
    # kẹp vẫn phải chạy theo ngón (trước đây kẹp đứng yên vì dùng chung độ tin cậy hướng bàn tay).
    cfg = load_config()
    cfg["mapping"]["robot_arms"] = ["right"]
    pipe = ShadowPipeline(cfg)
    pipe.seed({"right": np.zeros(8)})
    q = {"right": np.deg2rad([20, 20, 0, 60, 0, 0, 0])}
    for i in range(30):
        out = pipe.step(fake_frame(pipe, q, i / 15))
    g_open = out["right"][7]
    for i in range(30, 60):
        fr = fake_frame(pipe, q, i / 15)
        fr.arms["right"].grip = 0.2                     # chụm
        fr.arms["right"].conf.update(hand=0.2, grip=0.9)
        out = pipe.step(fr)
    assert out["right"][7] < g_open - 0.3 and not pipe.held["right"][7]
    assert pipe.held["right"][4]                         # cổ tay vẫn giữ vì hướng tay chưa chắc
    fr.arms["right"].conf.update(grip=0.3)               # chính 2 ngón không rõ -> kẹp giữ
    fr.arms["right"].grip = 0.9
    g = pipe.step(fr)["right"][7]
    assert np.isclose(g, out["right"][7]) and pipe.held["right"][7]
