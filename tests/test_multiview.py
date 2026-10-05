"""Fusion 2 camera: camera 0 trực diện, camera 1 lệch 45° (dữ liệu tổng hợp, không cần camera thật)."""
import types

import cv2
import numpy as np
import pytest

from openarm_shadow.core.geometry import rot, unit
from openarm_shadow.multiview import (CameraModel, HandOrientationTracker, MultiSample, MultiViewPerception,
                                      fuse_point, triangulate_weighted)
from openarm_shadow.perception import Frame, ArmObs, palm_frame_from_depth, body_frame

K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
SUBJECT = np.array([0.0, 0.0, 1.3])      # người đứng cách camera trực diện 1,3 m


def look_at(center, target, up=(0, -1, 0)):
    """Camera tại center nhìn về target; trả (R, t) với X_cam = R X + t (x phải, y xuống, z tới)."""
    z = unit(np.asarray(target, float) - center)
    x = unit(np.cross(np.asarray(up, float) * -1, z))
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    return R, -R @ center


def two_cams():
    c0 = CameraModel("front", K=K.copy(), dist=np.zeros(5))
    ang = np.deg2rad(45)
    C1 = SUBJECT + 1.3 * np.array([np.sin(ang), 0, -np.cos(ang)])
    R1, t1 = look_at(C1, SUBJECT)
    c1 = CameraModel("side45", R1, t1, K.copy(), np.zeros(5))
    return [c0, c1]


def observe(cams, X, noise_px=0.0, rng=np.random.default_rng(0)):
    out = []
    for c in cams:
        px = c.project(X[None])[0] + rng.normal(0, noise_px, 2)
        out.append((c, c.normalize(px[None])[0], 1.0))
    return out


def test_camera_setup_is_45_degrees():
    c0, c1 = two_cams()
    ang = np.degrees(np.arccos(c0.R[2] @ c1.R[2]))
    assert ang == pytest.approx(45, abs=0.5)
    assert np.allclose(c1.project(SUBJECT[None])[0], [320, 240], atol=1e-6)


def test_triangulation_exact_and_with_noise():
    cams = two_cams()
    rng = np.random.default_rng(1)
    errs = []
    for _ in range(200):
        X = SUBJECT + rng.uniform(-0.4, 0.4, 3)
        assert np.allclose(triangulate_weighted(observe(cams, X)), X, atol=1e-9)
        Xn, conf, info = fuse_point(observe(cams, X, 2.0, rng), [])
        assert info["views"] == 2
        errs.append(np.linalg.norm(Xn - X))
    assert np.sqrt(np.mean(np.square(errs))) < 0.012 and max(errs) < 0.04   # nhiễu MediaPipe ~2 px


def test_one_view_needs_depth():
    cams = two_cams()
    X = SUBJECT + [0.1, -0.2, 0.05]
    obs = observe(cams, X)[:1]
    assert fuse_point(obs, [])[0] is None
    Xd, conf, info = fuse_point(obs, [(cams[0], X + [0, 0, 0.005], 0.9)])
    assert np.linalg.norm(Xd - X) < 0.01 and info["depth"] == 1


def test_bad_view_is_overruled_by_depth():
    """Camera 45° đoán sai điểm 60 px lệch khỏi đường epipolar: sai số chiếu lại lộ ra, depth phân xử."""
    cams = two_cams()
    X = SUBJECT + [0.15, 0.1, -0.1]
    obs = observe(cams, X)
    c1, xy1, w1 = obs[1]
    obs[1] = (c1, xy1 + np.array([0.0, 60.0]) / 600.0, w1)
    X0, _, _ = fuse_point(obs, [])
    Xf, conf, info = fuse_point(obs, [(cams[0], X + [0, 0, 0.01], 0.9)])
    assert np.linalg.norm(Xf - X) < 0.02 < np.linalg.norm(X0 - X)
    assert conf < 1.0 and info["depth"] == 1


def test_error_along_epipolar_line_is_flagged():
    """Sai lệch DỌC đường epipolar (ngang ảnh với 2 camera đặt ngang nhau): 2 camera vẫn khớp nhau nên không
    phát hiện bằng chiếu lại được; depth lệch nghiệm -> báo xung đột và hạ độ tin cậy, không đoán bên nào đúng."""
    cams = two_cams()
    X = SUBJECT + [0.15, 0.1, -0.1]
    obs = observe(cams, X)
    c1, xy1, w1 = obs[1]
    obs[1] = (c1, xy1 + np.array([-60.0, 0.0]) / 600.0, w1)    # nghiệm 2D bị kéo về gần camera
    X2, _, _ = fuse_point(obs, [])
    assert cams[0].to_cam(X2)[2] < X[2] - 0.04
    _, conf, info = fuse_point(obs, [(cams[0], X + [0, 0, 0.01], 0.9)])
    assert info["conflict"] and conf <= 0.5


def test_occluder_depth_is_ignored_when_views_agree():
    """Depth ở pixel bị che = mặt phía trước (gần hơn 5 cm): hai camera khớp nhau thì bỏ depth đó."""
    cams = two_cams()
    X = SUBJECT + [0.0, 0.1, 0.0]
    Xf, conf, info = fuse_point(observe(cams, X), [(cams[0], X - [0, 0, 0.05], 0.9)])
    assert np.linalg.norm(Xf - X) < 1e-6 and info["depth"] == 0
    assert not info["conflict"] and conf > 0.9


# ---------------- bàn tay ----------------
def hand_points(side, R_hand, center):
    """21 điểm bàn tay trong khung bàn tay (x hướng ngón, y út->trỏ, lòng tay trong mặt z=0)."""
    base = {0: (0, 0), 5: (0.09, 0.035), 9: (0.095, 0.01), 13: (0.09, -0.012), 17: (0.08, -0.032),
            1: (0.03, 0.04), 2: (0.05, 0.06), 3: (0.065, 0.075), 4: (0.08, 0.085)}
    P = np.zeros((21, 3))
    for k, (x, y) in base.items():
        P[k] = [x, y, 0]
    for m in (5, 9, 13, 17):
        for j in range(1, 4):
            P[m + j] = P[m] + [0.025 * j, 0, 0]
    if side == "left":
        P[:, 1] *= -1
    return P @ R_hand.T + center


def test_palm_normal_no_flip_when_edge_on_to_front_camera():
    """Quay bàn tay quanh trục ngón qua cả tư thế nhìn cạnh từ camera trực diện: pháp tuyến không lật."""
    cams = two_cams()
    rng = np.random.default_rng(3)
    center = SUBJECT + [0.1, 0.1, -0.2]
    worst = 0.0
    for roll in np.arange(-180, 180, 7.5):
        # ngón chỉ sang phải (x camera), lăn quanh trục ngón; roll = +-90 => lòng tay song song tia nhìn camera 0
        R_hand = rot([1, 0, 0], np.deg2rad(roll)) @ np.column_stack([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        P = hand_points("right", R_hand, center)
        R_true, _ = palm_frame_from_depth(P, side="right")
        pts = []
        for j in range(21):
            X, _, _ = fuse_point(observe(cams, P[j], 1.5, rng), [])
            pts.append(X)
        R_est, _ = palm_frame_from_depth(np.array(pts), side="right")
        err = np.degrees(np.arccos(np.clip(R_est[:, 2] @ R_true[:, 2], -1, 1)))
        worst = max(worst, err)
    assert worst < 25.0, worst


def test_tracker_keeps_sign_when_all_views_edge_on():
    tr = HandOrientationTracker(alpha=1.0)
    R0 = np.eye(3)
    tr.update(R0, 1.0)
    flipped = R0 @ np.diag([1.0, -1.0, -1.0])
    R, mode = tr.update(flipped, 0.1)             # quan sát lật dấu, mọi camera nhìn cạnh
    assert np.allclose(R, R0) and mode == "SIGN-FIX"
    for _ in range(3):
        R, mode = tr.update(flipped, 0.9)         # nhìn rõ nhưng mới 1-3 khung: giữ
    assert mode == "HOLD"
    R, mode = tr.update(flipped, 0.9)             # lặp lại đủ 4 khung: nhận là xoay thật
    assert np.allclose(R, flipped)


# ---------------- end-to-end với MediaPipe giả ----------------
class FakeView:
    """Thay Perception: chiếu khung xương 3D thật vào camera, trả Frame như MediaPipe (toạ độ chuẩn hoá)."""

    def __init__(self, cam, world, hand, size=(640, 480), noise=1.0, seed=0):
        self.cam, self.world, self.hand, self.size = cam, world, hand, size
        self.rng = np.random.default_rng(seed)
        self.noise = noise

    def process(self, bgr, t=None):
        w, h = self.size
        px = self.cam.project(self.world) + self.rng.normal(0, self.noise, (33, 2))
        pose_2d = np.column_stack([px[:, 0] / w, px[:, 1] / h, np.full(33, 0.95)])
        hp = self.cam.project(self.hand) + self.rng.normal(0, self.noise, (21, 2))
        arms = {"right": ArmObs(), "left": ArmObs()}
        arms["right"].conf["hand"] = 0.95
        return Frame(arms, pose_2d, [(hp / [w, h], "right")], None, t or 0.0)


def human_world():
    """Người đứng đối diện camera 0; tay phải (bên trái ảnh) nâng ra trước, khuỷu gập."""
    W = np.tile(SUBJECT, (33, 1))
    W[11], W[12] = SUBJECT + [0.18, -0.35, 0], SUBJECT + [-0.18, -0.35, 0]     # vai trái / phải (x phải ảnh)
    W[23], W[24] = SUBJECT + [0.1, 0.15, 0], SUBJECT + [-0.1, 0.15, 0]         # hông
    W[13], W[15] = SUBJECT + [0.2, -0.08, 0.02], SUBJECT + [0.22, 0.18, 0.03]  # tay trái thả xuôi
    W[14] = W[12] + [0.0, 0.12, -0.26]                                          # khuỷu phải: tay trên ra trước
    W[16] = W[14] + [0.05, -0.18, -0.12]                                        # cổ tay phải: cẳng tay gập lên
    return W


def test_multiview_perception_end_to_end():
    cams = two_cams()
    W = human_world()
    R_hand = np.column_stack([unit([0.2, -0.8, -0.4]), unit([1, 0.3, 0.2]), [0, 0, 0]])
    R_hand[:, 1] = unit(R_hand[:, 1] - (R_hand[:, 1] @ R_hand[:, 0]) * R_hand[:, 0])
    R_hand[:, 2] = np.cross(R_hand[:, 0], R_hand[:, 1])
    hand = hand_points("right", R_hand, W[16])
    views = [FakeView(c, W, hand, seed=i) for i, c in enumerate(cams)]
    mvp = MultiViewPerception(views, cams, {"reproj_thresh_px": 25}, parallel=False)
    blank = types.SimpleNamespace(bgr=np.zeros((480, 640, 3), np.uint8), depth_m=None, intrinsics=None)
    fr = None
    for k in range(3):
        fr = mvp.process(MultiSample([blank, blank], 0.033 * k))
    ob = fr.arms["right"]
    Rb, origin = body_frame(W, np.ones(33))
    u_true, l_true = unit(Rb.T @ (W[14] - W[12])), unit(Rb.T @ (W[16] - W[14]))
    assert np.degrees(np.arccos(unit(ob.e - ob.s) @ u_true)) < 3
    assert np.degrees(np.arccos(unit(ob.w - ob.e) @ l_true)) < 3
    R_true, _ = palm_frame_from_depth(hand, side="right")
    assert np.degrees(np.arccos(np.clip((Rb @ ob.H)[:, 2] @ R_true[:, 2], -1, 1))) < 10
    assert fr.fusion["hand_right"]["views"] == 2
    # Kẹp: tỉ số đầu ngón cái-trỏ / bàn tay từ điểm 3D, độ tin cậy riêng
    tips = np.linalg.norm(hand[4] - hand[8]) / np.linalg.norm(hand[9] - hand[0])
    assert abs(ob.grip - tips) < 0.05 and ob.conf["grip"] >= 0.6


# ---------------- hiệu chuẩn ChArUco ----------------
def render_board(board, cam, R_b, t_b, size=(640, 480)):
    """Vẽ bảng phẳng vào camera pinhole bằng phép chiếu đồng nhất (homography)."""
    img_board = board.generateImage((700, 980), marginSize=0)
    sx, sy = board.getChessboardSize()
    sq = board.getSquareLength()
    Wm, Hm = sx * sq, sy * sq
    Hb = np.array([[Wm / 700, 0, 0], [0, Hm / 980, 0], [0, 0, 1]])        # pixel ảnh bảng -> mét trên bảng
    H = cam.K @ np.column_stack([R_b[:, 0], R_b[:, 1], t_b]) @ Hb
    return cv2.warpPerspective(img_board, H, size, borderValue=255)


def test_charuco_extrinsics_recovered():
    from openarm_shadow.calibration import (average_extrinsics, board_pose, detect, make_board,
                                            relative_extrinsic, reprojection_rms_px)
    board, det = make_board({"squares_x": 5, "squares_y": 7, "square_m": 0.035, "marker_m": 0.026})
    cams = two_cams()
    rng = np.random.default_rng(5)
    samples, dets = [], []
    for _ in range(8):
        # bảng giữa hai camera, nghiêng về phía camera 45°
        R_w = rot([0, 1, 0], np.deg2rad(rng.uniform(-35, -10))) @ rot([1, 0, 0], np.deg2rad(rng.uniform(-15, 15)))
        t_w = SUBJECT + rng.uniform(-0.1, 0.1, 3) - [0.08, 0.1, 0.3]
        poses, dd = [], []
        for c in cams:
            R_b, t_b = c.R @ R_w, c.R @ t_w + c.t
            img = render_board(board, c, R_b, t_b)
            ids, px = detect(det, img)
            dd.append((ids, px))
            poses.append(board_pose(board, c, ids, px))
        assert all(p is not None for p in poses)
        samples.append(relative_extrinsic(poses[0], poses[1]))
        dets.append(tuple(dd))
    R, t, keep = average_extrinsics(samples)
    ang = np.degrees(np.arccos(np.clip((np.trace(R.T @ cams[1].R) - 1) / 2, -1, 1)))
    assert ang < 0.5 and np.linalg.norm(t - cams[1].t) < 0.01
    est = CameraModel("side45", R, t, K.copy(), np.zeros(5))
    assert reprojection_rms_px(board, cams[0], est, dets) < 2.0


def test_webcam_calibration_size_is_checked(tmp_path):
    """Nội tham số webcam chỉ đúng ở độ phân giải lúc hiệu chuẩn: lệch thì dừng, không chạy với số sai."""
    import yaml
    from openarm_shadow.multiview import check_image_size, load_calibration
    f = tmp_path / "calib.yaml"
    f.write_text(yaml.safe_dump({"cameras": {
        "front": {"R": np.eye(3).tolist(), "t": [0, 0, 0], "K": K.tolist(), "dist": [0] * 5, "size": [640, 480]},
        "side45": {"R": np.eye(3).tolist(), "t": [0.5, 0, 0]}}}))
    front, side = load_calibration(f, ["front", "side45"])
    assert front.size == (640, 480) and side.size is None
    check_image_size(front, np.zeros((480, 640, 3), np.uint8))
    with pytest.raises(SystemExit):
        check_image_size(front, np.zeros((720, 1280, 3), np.uint8))
    side.set_realsense_intrinsics({"fx": 600, "fy": 600, "ppx": 320, "ppy": 240})
    check_image_size(side, np.zeros((720, 1280, 3), np.uint8))     # RealSense: nội tham số lấy từ SDK


def test_camera_stall_is_reported(monkeypatch):
    """Webcam ngừng gửi khung: read() trả False kèm lý do (camera nào, bao nhiêu khung/lỗi), không thoát im lặng."""
    import time as _time
    import openarm_shadow.sources as sources
    from openarm_shadow.multiview import MultiCameraSource

    class Fake:
        mode = "fake"

        def __init__(self, source, *a, **k):
            self.n, self.dead = 0, source == 0
        def read(self):
            _time.sleep(0.01)
            self.n += 1
            if self.dead and self.n > 3:
                raise RuntimeError("Frame didn't arrive")
            return True, types.SimpleNamespace(bgr=np.zeros((4, 4, 3), np.uint8), depth_m=None, intrinsics=None)
        def close(self):
            pass

    monkeypatch.setattr(sources, "OpenCVSource", Fake)
    cfg = {"fusion": {"cameras": [{"name": "front", "source": 0}, {"name": "side45", "source": 1}]},
           "camera": {"width": 640, "height": 480}}
    src = MultiCameraSource(cfg)
    try:
        ok, _ = src.read(timeout=1.0)
        assert ok
        oks = [src.read(timeout=0.3)[0] for _ in range(5)]
        assert not all(oks)
        assert "front" in src.error and "Frame didn't arrive" in src.error
    finally:
        src.close()


def test_frozen_second_camera_is_marked_stale(monkeypatch):
    """Camera phụ treo: không được ghép khung cũ của nó mãi mãi vào triangulation."""
    import time as _time
    import openarm_shadow.sources as sources
    from openarm_shadow.multiview import MultiCameraSource

    class Fake:
        mode = "fake"

        def __init__(self, source, *a, **k):
            self.n, self.freeze = 0, source == 1
        def read(self):
            _time.sleep(0.01)
            self.n += 1
            if self.freeze and self.n > 3:
                return False, None
            return True, types.SimpleNamespace(bgr=np.zeros((4, 4, 3), np.uint8), depth_m=None, intrinsics=None)
        def close(self):
            pass

    monkeypatch.setattr(sources, "OpenCVSource", Fake)
    cfg = {"fusion": {"cameras": [{"name": "front", "source": 0}, {"name": "side45", "source": 1}]},
           "camera": {"width": 640, "height": 480}}
    src = MultiCameraSource(cfg)
    try:
        _time.sleep(0.3)
        ok, ms = src.read(timeout=1.0)
        assert ok and ms.stale == [False, True]
    finally:
        src.close()
    # Perception bỏ qua camera stale (không gọi MediaPipe cho nó)
    cams = two_cams()
    calls = []

    class Spy(FakeView):
        def process(self, bgr, t=None):
            calls.append(self.cam.name)
            return super().process(bgr, t)
    W = human_world()
    views = [Spy(c, W, hand_points("right", np.eye(3), W[16])) for c in cams]
    mvp = MultiViewPerception(views, cams, {}, parallel=False)
    blank = types.SimpleNamespace(bgr=np.zeros((480, 640, 3), np.uint8), depth_m=None, intrinsics=None)
    fr = mvp.process(MultiSample([blank, blank], 0.0, 0.0, [False, True]))
    assert calls == ["front"] and fr.fusion["stale"] == ["side45"]


def test_stall_message_names_the_stalled_camera():
    from openarm_shadow.multiview import MultiCameraSource
    src = MultiCameraSource.__new__(MultiCameraSource)
    src.names = ["front", "side45"]
    src.stats = [{"frames": 90, "fails": 0, "error": None}, {"frames": 0, "fails": 3, "error": "timeout"}]
    assert src.describe_stall(3, "side45").startswith("Camera 'side45'")
    assert src.describe_stall(3).startswith("Camera 'front'")


def test_camera_time_offset_from_motion_signal():
    from openarm_shadow.calibration import estimate_time_offset
    rng = np.random.default_rng(3)
    t_ref = np.arange(0, 12, 1 / 30) + rng.uniform(0, 0.005, 360)
    bursts = lambda t: sum(np.exp(-((t - c) / 0.15) ** 2) for c in (2.0, 3.7, 5.1, 6.9, 8.2, 9.8))
    lag_true = 0.062                                     # iPhone đến máy chậm hơn webcam 62 ms
    t = np.arange(0.01, 12, 1 / 25)
    lag, corr = estimate_time_offset(t_ref, bursts(t_ref), t, bursts(t - lag_true) + rng.normal(0, 0.02, len(t)))
    assert abs(lag - lag_true) < 0.008 and corr > 0.9


def _source_with_buffers(sync, ref_times, other_times, latency=0.0):
    """MultiCameraSource không mở camera thật: đặt sẵn bộ đệm (thời điểm đã bù latency) rồi gọi read()."""
    import threading
    from collections import deque
    from openarm_shadow.multiview import MultiCameraSource
    src = MultiCameraSource.__new__(MultiCameraSource)
    src.names, src.tol, src.stale_s, src.pair_wait = ["front", "side45_left"], 0.025, 0.04, 0.0
    src.sync, src.sync_max_wait, src.last_ref_t, src.running = sync, 0.15, -1.0, True
    src.cond, src.error = threading.Condition(), None
    src.stats = [{"frames": 1, "fails": 0, "error": None}] * 2
    src.buf = [deque([(t, t) for t in ref_times], maxlen=12),
               deque([(t - latency, t) for t in other_times], maxlen=12)]
    return src


def test_sync_slowest_pairs_late_camera_at_same_moment():
    # Lúc 10.000 s: webcam đã có khung tới 9.999; iPhone (đến chậm 54 ms, bù latency_s 0.05) mới tới khung 9.950.
    ref = [10.0 - k / 30 for k in range(8)][::-1]
    late = [10.0 - 0.054 - k / 30 for k in range(8)][::-1]
    ok, ms = _source_with_buffers("latest", ref, late, 0.05).read(timeout=0.1)
    assert ok and ms.stale == [False, True]             # kiểu cũ: lấy khung webcam mới nhất -> iPhone bị bỏ
    src = _source_with_buffers("slowest", ref, late, 0.05)
    ok, ms = src.read(timeout=0.1)
    assert ok and ms.stale == [False, False] and ms.lags[1] < 0.025
    assert ms.t < 10.0 - 0.03                            # đổi lại: khung tham chiếu cũ hơn (trễ thêm)


def test_sync_slowest_does_not_wait_for_frozen_camera():
    ref = [10.0 - k / 30 for k in range(8)][::-1]
    frozen = [9.5]                                       # camera phụ treo từ 0,5 s trước
    ok, ms = _source_with_buffers("slowest", ref, frozen).read(timeout=0.1)
    assert ok and ms.t == ref[-1] and ms.stale == [False, True]
