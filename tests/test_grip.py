"""Kẹp: mở liên tục theo hai ngón, tuỳ chọn theo mức (có trễ + giữ), hiệu chuẩn theo người, mất tay thì giữ."""
import numpy as np

from openarm_shadow.config import load_config
from openarm_shadow.mapping.grip import GripMapper
from openarm_shadow.mapping.pipeline import ShadowPipeline
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


def _r(v):
    return 0.25 + v * 0.65


def test_grip_latch_ignores_brief_false_open():
    """run5 05/10: đang kẹp, tay di chuyển -> đo nhầm 'mở hẳn' 1-2 khung (0,1-0,25 s) -> kẹp nhả vật."""
    g = GripMapper(pinch_ratio=0.25, open_ratio=0.9, release_s=0.3)
    assert np.isclose(g(_r(0.0), 0.0), 0.0)
    assert np.isclose(g(_r(1.0), 0.1), 0.0) and g.releasing      # 1 khung nhầm: giữ kẹp
    assert np.isnan(g(None, 0.2))                                # mất tay: NaN, bộ lọc giữ
    assert np.isclose(g(_r(0.05), 0.3), 0.05) and not g.releasing   # quay về chụm: huỷ chờ
    for t in (1.0, 1.1, 1.2):                                    # mở 0,2 s: vẫn giữ
        assert np.isclose(g(_r(1.0), t), 0.05)
    assert np.isnan(g(None, 1.25)) and g.releasing               # mất tay giữa chừng không xoá đồng hồ
    assert np.isclose(g(_r(1.0), 1.35), 1.0)                     # mở liên tục >= 0,3 s, tay đứng yên: nhả
    assert np.isclose(g(_r(0.0), 1.45), 0.0)                     # đóng: nhận ngay


def test_grip_latch_off_and_partial_open():
    g = GripMapper(pinch_ratio=0.25, open_ratio=0.9)             # mặc định lớp: tắt, như cũ
    assert g(0.25, 0.0) == 0.0 and np.isclose(g(0.9, 0.1), 1.0)
    g = GripMapper(pinch_ratio=0.25, open_ratio=0.9, release_s=0.3)
    g(0.25, 0.0)
    assert np.isclose(g(_r(0.2), 0.1), 0.2)                      # nới nhẹ (< release_delta): nhận ngay


def test_grip_holds_while_carrying_object_and_releases_after_stop():
    """Mang vật: cổ tay đi 0,6 m/s, đo nhầm 'mở' 0,6 s liền (> release_s) -> vẫn kẹp. Dừng tay rồi mở -> nhả."""
    g = GripMapper(pinch_ratio=0.25, open_ratio=0.9, release_s=0.3, move_speed_mps=0.35, release_max_s=1.5)
    dt, x = 0.1, 0.0
    for k in range(10):                                          # đang kẹp, tay đi
        t = k * dt
        x += 0.06
        assert np.isclose(g(_r(0.0), t, wrist=[x, 0, 1]), 0.0)
    for k in range(10, 16):                                      # nhoè 0,6 s: đo 'mở' khi tay vẫn đi
        t = k * dt
        x += 0.06
        assert np.isclose(g(_r(1.0), t, wrist=[x, 0, 1]), 0.0)
    for k in range(16, 20):                                      # hết nhoè: chụm lại
        x += 0.06
        assert np.isclose(g(_r(0.0), k * dt, wrist=[x, 0, 1]), 0.0)
    out = [g(_r(0.0), k * dt, wrist=[x, 0, 1]) for k in range(20, 30)]   # dừng tay ở đích
    out += [g(_r(1.0), k * dt, wrist=[x, 0, 1]) for k in range(30, 36)]  # mở tay thả vật
    assert np.isclose(out[-1], 1.0)                              # tay đứng yên + mở 0,3 s: nhả


def test_grip_releases_eventually_when_opening_while_moving():
    g = GripMapper(pinch_ratio=0.25, open_ratio=0.9, release_s=0.3, move_speed_mps=0.35, release_max_s=1.5)
    g(_r(0.0), 0.0, wrist=[0, 0, 1])
    vals = [g(_r(1.0), k * 0.1, wrist=[0.06 * k, 0, 1]) for k in range(1, 20)]
    assert np.isclose(vals[12], 0.0) and np.isclose(vals[-1], 1.0)   # 1,3 s: còn kẹp; > 1,5 s: nhả


def test_grip_waits_longer_after_hand_reacquired_while_closed():
    """run6 06/10: đang kẹp, mất bàn tay vài khung, bắt lại MediaPipe đoán 'xoè' ổn định -> trước đây nhả sau 0,3 s."""
    g = GripMapper(pinch_ratio=0.25, open_ratio=0.9, release_s=0.3, release_reacquire_s=1.0)
    g(_r(0.0), 0.0)
    for t in (0.1, 0.2, 0.3):
        assert np.isnan(g(None, t))                              # mất tay
    vals = [g(_r(1.0), 0.4 + 0.1 * k) for k in range(12)]        # bắt lại: 'mở' ổn định
    assert np.allclose(vals[:9], 0.0)                            # < 1 s: còn kẹp
    assert np.isclose(vals[-1], 1.0)                             # mở liên tục > 1 s: nhả (người thả thật)
    g(_r(0.0), 2.0)
    assert not g._lost_latched
    assert np.isclose(g(_r(1.0), 2.1), 0.0) and np.isclose(g(_r(1.0), 2.45), 1.0)   # không mất tay: 0,3 s như cũ


def test_grip_release_needs_all_cameras_to_see_open():
    """1 camera đặt sai đầu ngón -> 3D 'mở', camera kia vẫn thấy chụm: giữ kẹp. Đóng vẫn theo giá trị fusion."""
    g = GripMapper(pinch_ratio=0.25, open_ratio=0.9, release_s=0.3)
    g(_r(0.0), 0.0, r_views=[_r(0.0), _r(0.0)])
    for k in range(1, 10):
        assert g(_r(1.0), 0.1 * k, r_views=[_r(1.0), _r(0.05)]) <= 0.1     # vẫn kẹp
    out = [g(_r(1.0), 1.0 + 0.1 * k, r_views=[_r(1.0), _r(0.95)]) for k in range(5)]
    assert np.isclose(out[-1], 1.0)                              # mọi camera thấy mở: nhả
    g2 = GripMapper(pinch_ratio=0.25, open_ratio=0.9, release_s=0.3)
    g2(_r(1.0), 0.0, r_views=[_r(1.0), _r(1.0)])
    assert np.isclose(g2(_r(0.5), 0.1, r_views=[_r(0.5), _r(0.0)]), 0.5)   # không kẹp: dùng fusion như cũ
