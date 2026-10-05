"""Bố cục cửa sổ: nhiều camera -> hình robot nằm dưới, rộng bằng dải camera; 1 camera -> bên phải như cũ."""
import numpy as np

from openarm_shadow.core.kinematics import ArmKinematics
from openarm_shadow.display.viz import compose_view, draw_robot

KINS = {"right": ArmKinematics("right")}
Q = {"right": np.zeros(8)}


def render(size, zoom):
    return draw_robot(KINS, Q, size=size, zoom_to_arms=zoom)


def test_multi_camera_robot_below():
    cam = np.zeros((360, 1280, 3), np.uint8)
    v = compose_view(cam, render)
    assert v.shape == (360 + 480, 1280, 3)


def test_single_camera_robot_right():
    cam = np.zeros((480, 640, 3), np.uint8)
    v = compose_view(cam, render)
    assert v.shape == (480, 640 + 480, 3)


def test_forced_layout_and_height():
    cam = np.zeros((480, 640, 3), np.uint8)
    assert compose_view(cam, render, layout="below", robot_height=300).shape == (780, 640, 3)


def test_zoom_draws_arm_bigger():
    def arm_pixels(img):
        return int(np.count_nonzero(np.any(img < 200, axis=2)))
    small = draw_robot(KINS, Q, size=(1280, 480))
    big = draw_robot(KINS, Q, size=(1280, 480), zoom_to_arms=True)
    assert arm_pixels(big) > 1.1 * arm_pixels(small)


def test_far_projected_points_are_skipped_not_crash():
    """Điểm 3D sát mặt phẳng camera chiếu ra toạ độ ~1e12 px: trước đây cv2.circle ném lỗi và dừng chương trình."""
    import types

    import cv2
    import pytest

    from openarm_shadow.fusion.multiview import MultiViewPerception
    from openarm_shadow.core.types import ArmObs, Frame
    from openarm_shadow.display.viz import draw_human, pixel

    img = np.zeros((480, 640, 3), np.uint8)
    assert pixel([10.4, 20.6], img) == (10, 21)
    assert pixel([np.nan, 5], img) is None and pixel([1e12, 5], img) is None
    with pytest.raises(cv2.error):                       # đúng lỗi đã gặp khi chạy 3 camera
        cv2.circle(img, (int(1e12), 5), 7, (255, 0, 255), 2)

    arms = {"right": ArmObs(), "left": ArmObs()}
    arms["right"].hand_axes_px = np.array([[100.0, 100.0], [1e12, 3e11], [120.0, 90.0], [np.nan, 0.0]])
    fr = Frame(arms, None, [], None, 0.0)
    draw_human(img.copy(), fr)                           # trục hỏng bị bỏ, trục tốt vẫn vẽ

    cam = types.SimpleNamespace(name="cam", project=lambda X: np.tile([[1e12, -4e11]], (len(X), 1)))
    stub = types.SimpleNamespace(cams=[cam])
    W = np.zeros((33, 3))
    sample = types.SimpleNamespace(views=[types.SimpleNamespace(bgr=img)])
    out = MultiViewPerception.draw(stub, sample, fr, view_frames=[fr], world=W)
    assert out.shape[0] == 360
