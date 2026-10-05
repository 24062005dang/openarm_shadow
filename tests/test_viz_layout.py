"""Bố cục cửa sổ: nhiều camera -> hình robot nằm dưới, rộng bằng dải camera; 1 camera -> bên phải như cũ."""
import numpy as np

from openarm_shadow.core.kinematics import ArmKinematics
from openarm_shadow.viz import compose_view, draw_robot

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
