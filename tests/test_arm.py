"""Lớp Arm (trái / phải dùng chung): quan hệ khớp trái-phải đọc từ URDF, ROS inference.md và MJCF; MIRROR_SIGNS."""
import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from openarm_shadow.mapping.arm import MIRROR_SIGNS, Arm, mirror_q, other_side
from openarm_shadow.config import ROOT, load_config
from openarm_shadow.core.kinematics import ArmKinematics
from openarm_shadow.mapping.pipeline import ShadowPipeline
from test_pipeline import fake_frame

MD = ROOT / "ROS inference.md"
MJCF = ROOT / "openarm_mujoco" / "v1" / "openarm_bimanual.xml"
R, L = ArmKinematics("right"), ArmKinematics("left")
M = np.diag([1.0, -1.0, 1.0])                         # phản chiếu qua mặt phẳng dọc giữa thân (y -> -y)


def md_limits():
    out = {}
    for m in re.finditer(r"\|\s*\d+\s*\|\s*`(left|right)_j(\d)`\s*\|\s*\d+\s*\|\s*([−\-\d.]+)\s*\|\s*([−\-\d.]+)\s*\|",
                         MD.read_text()):
        out[(m[1], int(m[2]))] = tuple(float(m[k].replace("−", "-")) for k in (3, 4))
    return out


def mjcf_joints():
    out = {}
    for j in ET.parse(MJCF).getroot().iter("joint"):
        m = re.match(r"openarm_(left|right)_joint(\d)$", j.get("name") or "")
        if m and j.get("range"):
            out[(m[1], int(m[2]))] = (tuple(map(float, j.get("range").split())), np.array(j.get("axis").split(), float))
    return out


@pytest.mark.skipif(not MD.exists(), reason="không có ROS inference.md")
def test_limits_urdf_equals_ros_doc_and_left_is_negated_only_for_j1_j2():
    doc = md_limits()
    if len(doc) != 14:
        pytest.skip("ROS inference.md không có bảng giới hạn từng khớp dạng mới (14 dòng left_jN / right_jN)")
    for k in range(1, 8):
        for kin, side in ((R, "right"), (L, "left")):
            assert np.allclose((kin.lower[k - 1], kin.upper[k - 1]), doc[(side, k)], atol=2e-4), (side, k)
        right, left = np.array(doc[("right", k)]), np.array(doc[("left", k)])
        if k in (1, 2):                                    # giới hạn ngược nhau
            assert np.allclose(left, -right[::-1], atol=2e-4) and not np.allclose(left, right, atol=1e-3)
        else:                                              # giống nhau (J3, J5, J6, J7 đối xứng quanh 0)
            assert np.allclose(left, right, atol=2e-4)


@pytest.mark.skipif(not MJCF.exists(), reason="chưa có openarm_mujoco/v1")
def test_mjcf_limits_and_axes_follow_the_same_relation():
    mj = mjcf_joints()
    for k in range(1, 8):
        (rl, rh), (ll, lh) = mj[("right", k)][0], mj[("left", k)][0]
        if k in (1, 2):
            assert np.allclose((ll, lh), (-rh, -rl), atol=1e-3)
        else:
            assert np.allclose((ll, lh), (rl, rh), atol=0.06)          # MJCF hẹp hơn URDF vài độ (J4 hai tay lệch ~3°)
        ax_r, ax_l = mj[("right", k)][1], mj[("left", k)][1]
        assert np.allclose(ax_l, -ax_r if k == 7 else ax_r)             # chỉ trục J7 tay trái ngược tay phải
        assert np.allclose(L.joints[k - 1].axis, ax_l) and np.allclose(R.joints[k - 1].axis, ax_r)


def rot_err(q):
    Rr, Rl = R.R0(q, 7), L.R0(MIRROR_SIGNS * q, 7)
    return np.degrees(np.arccos(np.clip((np.trace((M @ Rr @ M).T @ Rl) - 1) / 2, -1, 1)))


def test_mirror_signs_give_exact_mirror_of_every_link_orientation():
    rng = np.random.default_rng(0)
    for _ in range(200):
        q = rng.uniform(-1.2, 1.2, 7)
        assert rot_err(q) < 1e-3                           # độ (arccos mất chính xác gần 0)
        for i in range(1, 8):                              # mọi link, không chỉ link 7
            assert np.allclose(L.R0(MIRROR_SIGNS * q, i), M @ R.R0(q, i) @ M)
    assert list(MIRROR_SIGNS) == [-1, -1, -1, 1, -1, -1, -1]


def test_flipping_only_j1_j2_is_not_enough_and_mirror_q_is_an_involution():
    rng = np.random.default_rng(1)
    only12 = np.array([-1, -1, 1, 1, 1, 1, 1.0])
    worst = 0.0
    for _ in range(100):
        q = rng.uniform(-1.0, 1.0, 7)
        Rl = L.R0(only12 * q, 7)
        worst = max(worst, np.degrees(np.arccos(np.clip((np.trace((M @ R.R0(q, 7) @ M).T @ Rl) - 1) / 2, -1, 1))))
    assert worst > 90                                      # chỉ đảo J1, J2 làm cổ tay / cẳng tay sai chiều
    q8 = np.append(rng.uniform(-1, 1, 7), 0.7)
    q8[2] = np.nan
    twice = mirror_q(mirror_q(q8))
    assert np.allclose(twice, q8, equal_nan=True) and mirror_q(q8)[7] == 0.7 and np.isnan(mirror_q(q8)[2])
    assert other_side("left") == "right"


def test_mirrored_joint_limits_are_consistent_with_mirror_signs():
    for k in range(7):
        lo, hi = sorted((MIRROR_SIGNS[k] * R.lower[k], MIRROR_SIGNS[k] * R.upper[k]))
        if k in (0, 1):                                    # J1, J2: dải tay phải đảo dấu = dải tay trái
            assert np.allclose((lo, hi), (L.lower[k], L.upper[k]), atol=1e-6)
        elif k == 3:
            assert np.allclose((lo, hi), (L.lower[k], L.upper[k]), atol=1e-6)
        else:                                              # đối xứng quanh 0: đảo dấu không đổi dải
            assert np.allclose((lo, hi), (L.lower[k], L.upper[k]), atol=1e-6)


def test_arm_objects_and_backward_compatible_side_views():
    cfg = load_config()
    pipe = ShadowPipeline(cfg)
    assert set(pipe.arms) == {"right", "left"} and all(isinstance(a, Arm) for a in pipe.arms.values())
    assert pipe.arms["left"].kin.side == "left" and pipe.arms["left"].partner == "right"
    assert pipe.kins["right"] is pipe.arms["right"].kin and pipe.rt["left"] is pipe.arms["left"].rt
    pipe.hand_calibrated["right"] = True                   # gán qua view ghi vào Arm
    assert pipe.arms["right"].hand_calibrated and not pipe.arms["left"].hand_calibrated
    pipe.q_prev["left"] = np.ones(7)
    assert np.allclose(pipe.arms["left"].q_prev, 1.0)
    assert "right" not in pipe.last_info and pipe.last_info.get("right") is None
    assert pipe.lm_kf is None and pipe.arm_shape is None
    cfg["filter"]["landmark_kalman"] = {"enabled": True}
    assert ShadowPipeline(cfg).lm_kf["left"] is not None


def test_mirror_mode_equals_sign_flipped_right_arm():
    """Người điều khiển tay phải: chế độ gương cho tay trái robot đúng MIRROR_SIGNS * (góc tay phải robot chế độ direct)."""
    cfg = load_config()
    cfg["mapping"]["robot_arms"] = ["right"]
    direct = ShadowPipeline(cfg)
    direct.seed({"right": np.zeros(8)})
    q_true = {"right": np.deg2rad([35, 20, 15, 70, 20, 15, 25])}
    out_r = None
    for i in range(60):
        out_r = direct.step(fake_frame(direct, q_true, i / 30))["right"]
    cfg2 = load_config()
    cfg2["mapping"].update(robot_arms=["left"], mode="mirror")
    mirror = ShadowPipeline(cfg2)
    mirror.seed({"left": np.zeros(8)})
    assert mirror.arms["left"].mirrored and mirror.arms["left"].human_side == "right"
    out_l = None
    for i in range(60):
        fr = fake_frame(direct, q_true, i / 30)            # cùng quan sát tay PHẢI người
        fr.arms["right"], fr.arms["left"] = fr.arms["right"], fr.arms["right"]
        out_l = mirror.step(fr)["left"]
    assert np.allclose(out_l[:7], MIRROR_SIGNS * out_r[:7], atol=np.deg2rad(2.0)), (np.rad2deg(out_l[:7]), np.rad2deg(out_r[:7]))
