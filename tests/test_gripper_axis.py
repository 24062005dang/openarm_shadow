"""Hướng kẹp: hai ngón kẹp đóng / mở theo PHÁP TUYẾN lòng bàn tay (mặt phẳng hai ngón kẹp vuông góc lòng bàn tay, như
ngón cái - ngón trỏ khi gắp). Trước đây hiệu chuẩn (lòng bàn tay nhìn camera) gán J5-J7 = 0 -> kẹp lệch 90°."""
import numpy as np
import pytest

from openarm_shadow.config import load_config
from openarm_shadow.core.types import ArmObs, Frame
from openarm_shadow.mapping.pipeline import ShadowPipeline
from openarm_shadow.vision.depth import palm_frame_from_depth
from test_multiview import hand_points

X, Y, Z = np.eye(3)
DOWN, FWD = -Z, X
THUMB_OUT = {"right": -Y, "left": Y}                  # tay xuôi, lòng bàn tay ra trước: ngón cái chỉ ra ngoài


def H_of(side, finger, thumb):
    """Khung bàn tay như perception đo (21 điểm -> palm_frame_from_depth): ngón theo `finger`, ngón cái về `thumb`."""
    side_axis = thumb if side == "right" else -thumb   # hand_points tay trái lật trục y cục bộ
    R = np.column_stack([finger, side_axis, np.cross(finger, side_axis)])
    return palm_frame_from_depth(hand_points(side, R, np.zeros(3)), side=side)[0]


def obs(side, l, H):
    s = np.array([0.0, -0.2 if side == "right" else 0.2, 0.0])
    e = s + 0.3 * DOWN
    return ArmObs(s=s, e=e, w=e + 0.27 * l, H=H, conf={"upper": 1.0, "fore": 1.0, "hand": 1.0})


@pytest.mark.parametrize("mode", ["direct", "mirror"])
def test_gripper_closes_along_palm_normal_after_calibration(mode):
    cfg = load_config()
    cfg["mapping"]["mode"] = mode
    pipe = ShadowPipeline(cfg)
    hs = {s: pipe.human_side_for(s) for s in pipe.robot_sides}
    # hiệu chuẩn như app: tay xuôi, lòng bàn tay nhìn camera
    pipe.calibrate_hand_neutral(Frame({h: obs(h, DOWN, H_of(h, DOWN, THUMB_OUT[h])) for h in ("right", "left")},
                                      None, [], np.eye(3), 0.0))
    poses = [(DOWN, lambda h: THUMB_OUT[h], 90), (DOWN, lambda h: FWD, 0), (FWD, lambda h: Z, 0),
             (FWD, lambda h: THUMB_OUT[h], 90), (FWD, lambda h: -THUMB_OUT[h], -90)]
    for s in pipe.robot_sides:
        h = hs[s]
        sign = 1.0 if s == "right" else -1.0          # J5 tay trái ngược dấu (MIRROR_SIGNS)
        for l, thumb, j5 in poses:
            fr = Frame({h: obs(h, l, H_of(h, l, thumb(h)))}, None, [], np.eye(3), 0.0)
            ob = pipe._obs_for_robot(fr, s)
            q, _ = pipe.rt[s].solve(ob.e - ob.s, ob.w - ob.e, ob.H, np.zeros(7))
            jaw = pipe.kins[s].R0(q, 7) @ np.array([0.0, -1.0, 0.0])     # trục đóng / mở ngón kẹp = -y link7
            human_normal = fr.arms[h].H[:, 2] * ([1, -1, 1] if mode == "mirror" else 1)
            assert abs(jaw @ human_normal) > 0.999, (mode, s, np.rad2deg(q[4:]))
            assert np.isclose(np.rad2deg(q[4]), sign * j5, atol=1.0), (mode, s, np.rad2deg(q[4:]))
            assert np.allclose(np.rad2deg(q[5:7]), 0.0, atol=1.0)


def test_calibration_wrist_reference():
    pipe = ShadowPipeline(load_config())
    assert np.allclose(np.rad2deg(pipe.arms["right"].calib_wrist), [90, 0, 0], atol=1e-3)   # URDF làm tròn ±1.570796
    assert np.allclose(np.rad2deg(pipe.arms["left"].calib_wrist), [-90, 0, 0], atol=1e-3)
