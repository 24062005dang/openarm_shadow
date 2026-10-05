"""Sửa theo lần chạy robot thật đầu (tri_real1): J3 trôi khi tay gần thẳng, điểm sai số lớn, cổ tay lật ±87°."""
import numpy as np

from openarm_shadow.config import load_config
from openarm_shadow.filtering import JointFilter
from openarm_shadow.fusion import err_conf_factor
from openarm_shadow.pipeline import ShadowPipeline
from test_pipeline import fake_frame


def test_err_conf_factor():
    assert err_conf_factor(5.0) == 1.0 and err_conf_factor(np.nan) == 1.0
    assert abs(err_conf_factor(18.0) - 0.6) < 1e-9
    assert err_conf_factor(80.0) == 0.3


def run_pipe(j4_deg):
    cfg = load_config()
    cfg["mapping"]["robot_arms"] = ["right"]
    pipe = ShadowPipeline(cfg)
    pipe.seed({"right": np.zeros(8)})
    q = {"right": np.deg2rad([20, 20, 30, j4_deg, 0, 0, 0])}
    for i in range(30):
        pipe.step(fake_frame(pipe, q, i / 15))
    return pipe


def test_j3_confidence_ramps_with_elbow_bend():
    near_straight, mid, bent = run_pipe(18), run_pipe(24), run_pipe(60)
    c = lambda p: p.conf["right"][2]
    assert c(near_straight) < 0.6 <= 0.9 * c(bent)        # gần thẳng: J3 giữ
    assert c(near_straight) < c(mid) < c(bent)
    assert abs(c(bent) - 0.9) < 1e-9                       # gập rõ: không ảnh hưởng
    assert near_straight.held["right"][2] and not bent.held["right"][2]


def test_wrist_big_jump_needs_tracking_confidence():
    cfg = load_config()["filter"]
    jf = JointFilter(8, cfg["min_cutoff"], cfg["beta"], cfg["deadband_deg"], cfg["jump_deg"], cfg["jump_hold_s"],
                     cfg["min_conf"], angular=[True] * 7 + [False], jump_confirm_conf=cfg["jump_confirm_conf"])
    x = np.zeros(8)
    for k in range(10):
        jf(x, np.full(8, 0.9), k * 0.1)
    flip = x.copy()
    flip[4] = np.deg2rad(170)                              # J5 lật -87 -> 87
    conf = np.full(8, 0.9)
    conf[4:7] = 0.7                                        # DEGRADED: 1 nguồn hướng tay
    for k in range(20):
        out, held = jf(flip, conf, 1.0 + 0.1 * k)
    assert held[4] and abs(out[4]) < 0.01
    conf[4:7] = 0.9                                        # TRACKING: >= 2 nguồn đồng ý -> nhận sau jump_hold_s
    for k in range(5):
        out, held = jf(flip, conf, 3.0 + 0.1 * k)
    assert not held[4]
