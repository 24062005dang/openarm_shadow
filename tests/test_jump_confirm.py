"""Chống lật J1/J3 (khuỷu nhảy trước/sau thân): bước nhảy chỉ nhận khi giá trị mới ổn định và đáng tin."""
import subprocess
import sys
from pathlib import Path

import numpy as np

from openarm_shadow.filtering import JointFilter
from openarm_shadow.fusion import fuse_point


def make():
    return JointFilter(1, 5.0, 0.0, [0.0], jump_deg=35, jump_hold_s=0.2, min_conf=0.6, jump_confirm_conf=0.7)


def settle(jf):
    for k in range(10):
        jf(np.array([0.0]), np.array([0.9]), k * 0.1)
    return 1.0


def test_jump_with_good_conf_accepted_after_hold():
    jf = make()
    t = settle(jf)
    for k in range(4):
        out, held = jf(np.array([1.5]), np.array([0.9]), t + 0.1 * k)
    assert not held[0] and out[0] > 1.0


def test_jump_with_weak_conf_never_accepted():
    jf = make()
    t = settle(jf)
    for k in range(20):                                   # 2 s ở hướng mới nhưng chỉ tin cậy 0.65 (1 camera + depth)
        out, held = jf(np.array([1.5]), np.array([0.65]), t + 0.1 * k)
    assert held[0] and abs(out[0]) < 0.01


def test_flickering_jump_not_accepted():
    jf = make()
    t = settle(jf)
    for k in range(20):                                   # nhảy qua lại giữa hai nghiệm lật: không ổn định
        x = 1.5 if k % 2 == 0 else -1.5
        out, held = jf(np.array([x]), np.array([0.9]), t + 0.1 * k)
        assert held[0]


def test_small_moves_unaffected():
    jf = make()
    t = settle(jf)
    out, held = jf(np.array([0.3]), np.array([0.65]), t)  # 17° < jump_deg: nhận ngay dù tin cậy vừa đủ
    assert not held[0]


class Cam:
    def to_cam(self, X):
        return np.asarray(X, float)


def test_body_mono_depth_conf_scaled():
    cam = Cam()
    obs = [(cam, np.array([0.0, 0.0]), 0.95)]
    dep = [(cam, np.array([0.0, 0.0, 1.0]), 0.9)]
    _, c_hand, _ = fuse_point(obs, dep)
    _, c_body, _ = fuse_point(obs, dep, mono_depth_conf=0.5)
    assert c_hand > 0.7 and c_body < 0.6


def test_find_jumps_script(tmp_path):
    t = np.arange(0, 5, 0.1)
    g = np.zeros((len(t), 8))
    g[30:, 0] = np.deg2rad(90)
    conf = np.full((len(t), 8), 0.9)
    fus = np.tile([2, 1, 2, 3.0, 8.0, 4.0, 0, 1, 0], (len(t), 1))
    f = tmp_path / "r.npz"
    np.savez(f, t=t, target_right=g, conf_right=conf, fus_right=fus)
    root = Path(__file__).resolve().parents[1]
    out = subprocess.run([sys.executable, str(root / "scripts" / "find_jumps.py"), str(f)],
                         capture_output=True, text=True, check=True).stdout
    assert "J1:    0.0 ->   90.0" in out and "khuyu 1c 8px D" in out and "=> 1 lần" in out


def test_fusion_row_reads_arm_points():
    from types import SimpleNamespace

    from openarm_shadow.app import fusion_row
    from openarm_shadow.perception import ARM_IDX
    s, e, w = ARM_IDX["right"]
    fr = SimpleNamespace(fusion={"points": {s: {"views": 2, "err_px": 3.0, "depth": 0},
                                            e: {"views": 1, "err_px": np.nan, "depth": 1}}})
    row = fusion_row(fr, "right")
    assert row[0] == 2 and row[1] == 1 and np.isnan(row[2]) and row[3] == 3.0 and row[7] == 1
    assert np.all(np.isnan(fusion_row(SimpleNamespace(fusion={}), "right")))
