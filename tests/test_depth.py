import numpy as np
import pytest

from openarm_shadow.perception import (deproject_pixel, fit_metric_depth, fuse_hand_landmarks,
                                       fit_palm_plane, open_finger_count, palm_frame_from_depth,
                                       rotation_distance, sample_depth, sample_depth_with_confidence,
                                       slerp_rotation)


def test_deproject_center_and_offset():
    intr = {"fx": 500.0, "fy": 400.0, "ppx": 320.0, "ppy": 240.0}
    assert np.allclose(deproject_pixel(intr, 320, 240, 2.0), [0, 0, 2])
    assert np.allclose(deproject_pixel(intr, 370, 200, 2.0), [0.2, -0.2, 2])


def test_sample_depth_uses_local_cluster_not_background():
    depth = np.full((11, 11), 3.0, dtype=np.float32)
    depth[4:7, 4:7] = 1.2
    depth[5, 5] = 1.21
    assert sample_depth(depth, 5, 5, radius=3, max_delta_m=0.15) == pytest.approx(1.2, abs=0.02)


def test_sample_depth_finds_nearest_when_center_is_hole():
    depth = np.zeros((9, 9), dtype=np.float32)
    depth[3:6, 3:6] = 1.5
    depth[4, 4] = 0
    assert sample_depth(depth, 4, 4, radius=2) == pytest.approx(1.5)


def test_sample_depth_rejects_invalid_range():
    depth = np.full((5, 5), 8.0, dtype=np.float32)
    assert sample_depth(depth, 2, 2, max_m=6.0) is None


def test_sample_depth_reports_quality():
    depth = np.full((9, 9), 1.25, dtype=np.float32)
    z, confidence = sample_depth_with_confidence(depth, 4, 4, radius=2)
    assert z == pytest.approx(1.25)
    assert confidence > 0.95


def test_fit_metric_depth_rejects_outlier():
    relative = np.linspace(-0.1, 0.1, 9)
    measured = 0.8 * relative + 1.2
    measured[4] = 2.5
    model = fit_metric_depth(relative, measured, np.ones(9), min_points=4)
    assert model[0] == pytest.approx(0.8, abs=0.05)
    assert model[1] == pytest.approx(1.2, abs=0.02)


def test_fit_metric_depth_uses_wrist_anchor_when_sparse():
    relative = np.array([0.0, -0.02, -0.04])
    measured = np.array([1.4, np.nan, np.nan])
    assert fit_metric_depth(relative, measured, min_points=4) == pytest.approx((1.4, 1.4))


class _Landmark:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


def test_fused_hand_points_stay_in_camera_metric_frame():
    depth = np.full((100, 100), 1.0, dtype=np.float32)
    # Mô phỏng lỗ stereo ở nửa số landmark; model vẫn phải dựng đủ 21 điểm trong D455 frame.
    landmarks = [_Landmark(0.3 + 0.01 * i, 0.5, -0.002 * i) for i in range(21)]
    for i, p in enumerate(landmarks):
        if i % 2:
            x, y = int(round(p.x * 100)), int(round(p.y * 100))
            depth[y, x] = 0.0
    intr = {"fx": 100.0, "fy": 100.0, "ppx": 50.0, "ppy": 50.0}
    cfg = {"hand_patch_radius": 0, "min_depth_m": 0.3, "max_depth_m": 3.0,
           "max_local_delta_m": 0.15, "hand_min_direct_points": 4}
    points, info = fuse_hand_landmarks(landmarks, depth, intr, cfg)
    assert info["direct"] >= 4
    assert info["fused"] == 21
    assert np.all(np.isfinite(points))
    assert np.all((points[:, 2] > 0.3) & (points[:, 2] < 3.0))


def _open_hand_points():
    p = np.full((21, 3), np.nan)
    p[0] = [0.0, 0.0, 1.0]
    # Bốn ngón thẳng theo +x, trải từ index (y âm) tới pinky (y dương).
    for chain, yy, base_x in zip(((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)),
                                 (-0.035, -0.012, 0.015, 0.04), (0.07, 0.08, 0.078, 0.065)):
        for j, idx in enumerate(chain):
            p[idx] = [base_x + 0.03 * j, yy, 1.0]
    return p


def test_open_hand_and_palm_frame_are_well_formed():
    points = _open_hand_points()
    assert open_finger_count(points) == 4
    R, center = palm_frame_from_depth(points, np.array([0.0, 0.0, 1.0]), 0.8)
    assert center.shape == (3,)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-6)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-6)
    # Plane normal phải được đổi dấu để nhất quán với landmark normal, không lật frame 180 độ.
    assert R[2, 2] < -0.9


def test_right_hand_chirality_flips_palm_normal_outward():
    points = _open_hand_points()
    left, _ = palm_frame_from_depth(points, side="left")
    right, _ = palm_frame_from_depth(points, side="right")
    assert left[:, 2] @ right[:, 2] < -0.99
    assert np.linalg.det(right) == pytest.approx(1.0, abs=1e-6)


def test_fit_palm_plane_from_depth_roi():
    depth = np.full((120, 120), 1.0, dtype=np.float32)
    coords = [(0.50, 0.75), (0.35, 0.38), (0.50, 0.32), (0.60, 0.36), (0.70, 0.42)]
    landmarks = [_Landmark(0.5, 0.5, 0.0) for _ in range(21)]
    for idx, xy in zip((0, 5, 9, 13, 17), coords):
        landmarks[idx] = _Landmark(*xy, 0.0)
    intr = {"fx": 120.0, "fy": 120.0, "ppx": 60.0, "ppy": 60.0}
    normal, center, info = fit_palm_plane(
        depth, intr, landmarks,
        {"min_depth_m": 0.3, "max_depth_m": 3.0, "palm_min_points": 35,
         "palm_depth_band_m": 0.06, "palm_plane_max_rms_m": 0.012})
    assert info["inliers"] >= 35
    assert info["rms_m"] < 1e-5
    assert center[2] == pytest.approx(1.0)
    assert abs(normal[2]) > 0.99


def test_slerp_rotation_has_expected_half_angle():
    R0 = np.eye(3)
    R1 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    halfway = slerp_rotation(R0, R1, 0.5)
    assert np.rad2deg(rotation_distance(R0, halfway)) == pytest.approx(45.0, abs=1e-6)
    assert np.allclose(halfway.T @ halfway, np.eye(3), atol=1e-6)


def _fake_perception():
    """Perception không cần file model: thay detector MediaPipe bằng kết quả dựng sẵn."""
    import types
    from openarm_shadow.perception import Perception
    p = Perception.__new__(Perception)
    p.mp = types.SimpleNamespace(Image=lambda **k: None, ImageFormat=types.SimpleNamespace(SRGB=0))
    p._last_ts = -1
    p.depth_cfg, p.orientation_cfg = {}, {}
    for name in ("_hand_depth_model", "_hand_orientation"):
        setattr(p, name, {"right": None, "left": None})
    p._hand_depth_misses = {"right": 0, "left": 0}
    p._orientation_tracking = {"right": False, "left": False}
    p._orientation_good = {"right": 0, "left": 0}
    p._orientation_bad = {"right": 0, "left": 0}
    p._body_R = None
    return p


def test_webcam_without_depth_still_gives_hand_orientation():
    """Không có D455: H (J5–J7) phải lấy từ điểm world của MediaPipe Hand, không được bỏ trống."""
    import types
    L = lambda x, y, z=0.0, v=0.99: types.SimpleNamespace(x=x, y=y, z=z, visibility=v)
    pose2d = [L(0.5, 0.5) for _ in range(33)]
    world = [L(0, 0, 0) for _ in range(33)]
    # vai trái/phải, khuỷu, cổ tay, hông (world: x phải, y xuống, z xa camera; người nhìn camera)
    for i, (x, y) in {11: (0.6, 0.3), 12: (0.4, 0.3), 13: (0.62, 0.45), 14: (0.38, 0.45),
                      15: (0.63, 0.6), 16: (0.37, 0.6), 23: (0.57, 0.7), 24: (0.43, 0.7)}.items():
        pose2d[i] = L(x, y)
    for i, (x, y) in {11: (0.18, -0.45), 12: (-0.18, -0.45), 13: (0.2, -0.17), 14: (-0.2, -0.17),
                      15: (0.21, 0.08), 16: (-0.21, 0.08), 23: (0.1, 0.0), 24: (-0.1, 0.0)}.items():
        world[i] = L(x, y, 0.0)
    # bàn tay phải thả xuôi, ngón chỉ xuống, lòng bàn tay nhìn camera
    hw = np.zeros((21, 3))
    for k, (dx, dy) in {0: (0, 0), 5: (-0.035, 0.09), 9: (-0.01, 0.095), 13: (0.012, 0.09),
                        17: (0.032, 0.08)}.items():
        hw[k] = [dx, dy, 0]
    for m in (5, 9, 13, 17):
        for j in range(1, 4):
            hw[m + j] = hw[m] + [0, 0.025 * j, 0]
    hw[1:5] = [[-0.04, 0.03, 0], [-0.06, 0.05, 0], [-0.075, 0.065, 0], [-0.085, 0.08, 0]]
    h2 = [L(0.37 + p[0], 0.6 + p[1]) for p in hw]
    hres = types.SimpleNamespace(hand_landmarks=[h2], hand_world_landmarks=[[L(*p) for p in hw]],
                                 handedness=[[types.SimpleNamespace(score=0.95)]])
    pres = types.SimpleNamespace(pose_landmarks=[pose2d], pose_world_landmarks=[world])
    p = _fake_perception()
    p.pose = types.SimpleNamespace(detect_for_video=lambda img, ts: pres)
    p.hands = types.SimpleNamespace(detect_for_video=lambda img, ts: hres)
    img = np.zeros((480, 640, 3), np.uint8)
    p.process(img, t=0.0)                       # khung đầu: bộ ổn định chờ nhận lại hướng
    fr = p.process(img, t=0.033)
    ob = fr.arms["right"]
    assert ob.H is not None and ob.hand_R_cam is not None
    assert np.allclose(ob.H.T @ ob.H, np.eye(3), atol=1e-6)
    assert ob.hand_open_fingers == 4
    assert ob.hand_R_cam[2, 2] < -0.9            # lòng bàn tay nhìn camera -> pháp tuyến hướng về camera (-z)
