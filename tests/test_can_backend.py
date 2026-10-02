"""Kiểm tra backend CAN với openarm_can giả lập: lọc số đọc rác, 2 lần đọc khớp nhau, dừng khi hỏng liên tục."""
import sys
import time
import types

import numpy as np
import pytest

from openarm_shadow.config import load_config

GARBAGE = -12.4676   # giá trị giải mã từ phản hồi 0x55 của Damiao đã gặp ở taichi_player


def make_fake_openarm_can():
    oa = types.ModuleType("openarm_can")

    class _E:
        def __getattr__(self, k):
            return k

    oa.MotorType = _E()
    oa.CallbackMode = _E()

    class MITParam:
        def __init__(self, kp, kd, q, dq, tau):
            self.kp, self.kd, self.q = kp, kd, q

    class _Stats:
        def __init__(self, m):
            self.m = m

        def seconds_since_response(self):
            return time.monotonic() - self.m.t

    class _Motor:
        def __init__(self, q):
            self.q, self.t, self.glitch = q, time.monotonic(), []   # glitch: các giá trị rác sẽ trả về lần tới

        def get_position(self):
            return self.glitch.pop(0) if self.glitch else self.q

    class _Comp:
        def __init__(self, qs):
            self.ms = [_Motor(q) for q in qs]

        def get_motors(self):
            return self.ms

        def get_link_stats(self, i):
            return _Stats(self.ms[i])

        def mit_control_all(self, ps):
            for m, p in zip(self.ms, ps):
                if p.kp > 0:
                    m.q += 0.5 * (p.q - m.q)
                m.t = time.monotonic()

    class OpenArm:
        instances = []

        def __init__(self, iface, fd):
            self.a, self.g, self.enabled = None, _Comp([0.0]), False
            OpenArm.instances.append(self)

        def init_arm_motors(self, t, a, b):
            self.a = _Comp([0.02, 0.05, 0.0, 0.1, 0.0, 0.0, 0.3])

        def init_gripper_motor(self, *a):
            pass

        def set_callback_mode_all(self, m):
            pass

        def refresh_all(self):
            for m in self.a.ms + self.g.ms:
                m.t = time.monotonic()

        def recv_all(self, t=500):
            pass

        def enable_all(self):
            self.enabled = True

        def disable_all(self):
            self.enabled = False

        def get_arm(self):
            return self.a

        def get_gripper(self):
            return self.g

    oa.MITParam, oa.OpenArm = MITParam, OpenArm
    return oa


@pytest.fixture
def robot(monkeypatch):
    oa = make_fake_openarm_can()
    monkeypatch.setitem(sys.modules, "openarm_can", oa)
    from openarm_shadow.robot.openarm_can_robot import OpenArmCANRobot
    cfg = load_config()
    cfg["robot"]["feedback_timeout_s"] = 1.0   # máy test bận không được làm test hỏng vì "mất phản hồi" giả
    r = OpenArmCANRobot(cfg["robot"], ["right"])
    return r, oa.OpenArm.instances[-1]


def test_connect_ignores_single_garbage_read(robot):
    r, hw = robot
    hw.a.ms[0].glitch = [GARBAGE]
    q = r.connect()
    assert q["right"][0] == pytest.approx(0.02)


def test_connect_fails_when_reads_never_agree(robot):
    from openarm_shadow.robot.openarm_can_robot import RobotFault
    r, hw = robot
    hw.a.ms[1].glitch = [0.05 + 0.1 * (i % 2) for i in range(100)]   # dao động 0.1 rad mỗi lần đọc
    with pytest.raises(RobotFault):
        r.connect()


def test_send_holds_last_good_value_on_single_glitch(robot):
    r, hw = robot
    q0 = r.connect()["right"]
    r.enable()
    hw.a.ms[3].glitch = [GARBAGE]
    r.send({"right": q0})
    assert np.isfinite(r.arms["right"].q_motor).all()
    assert abs(r.read()["right"][3] - q0[3]) < 0.05
    assert r.rejected_reads()["right"] == 1
    r.send({"right": q0})                  # lần sau đọc tốt -> hết trạng thái hỏng
    assert np.isnan(r.arms["right"].bad_since).all()


def test_persistent_garbage_faults_but_not_while_returning(robot):
    from openarm_shadow.robot.openarm_can_robot import RobotFault
    r, hw = robot
    q0 = r.connect()["right"]
    r.enable()
    hw.a.ms[2].glitch = [GARBAGE] * 1000
    r.returning = True
    for _ in range(30):                    # 0.3 s đọc hỏng khi đang về: không dừng
        r.send({"right": q0})
        time.sleep(0.01)
    r.returning = False
    with pytest.raises(RobotFault):
        r.send({"right": q0})


def test_enable_refuses_if_pose_changed(robot):
    from openarm_shadow.robot.openarm_can_robot import RobotFault
    r, hw = robot
    r.connect()
    hw.a.ms[0].q += np.deg2rad(10)         # tay bị cầm di chuyển sau khi đọc
    with pytest.raises(RobotFault):
        r.enable()
    assert not hw.enabled


def test_poll_reads_without_enabling(robot):
    r, hw = robot
    r.connect()
    hw.a.ms[0].q = 0.1                     # người cầm tay robot nâng J1 lên (motor tắt)
    assert r.poll()["right"][0] == pytest.approx(0.1) and not hw.enabled
    hw.a.ms[0].q = 0.6                     # nhảy lớn giữa hai lần đọc: lần đầu bị nghi là rác...
    assert r.poll()["right"][0] == pytest.approx(0.1)
    r.poll()
    assert r.poll()["right"][0] == pytest.approx(0.6)   # ...đọc giống nhau 3 lần thì chấp nhận


def test_poll_tolerates_slow_camera_loop_but_not_silent_joint(robot, monkeypatch):
    """--dry-run hỏi góc mỗi khung camera (80-150 ms): J1 trả lời trễ 0,2 s không phải mất phản hồi; im 1 s thì là."""
    from openarm_shadow.robot.openarm_can_robot import RobotFault
    r, hw = robot
    r.connect()
    j1 = hw.a.ms[0]
    real_refresh = type(hw).refresh_all

    def refresh_without_j1(self, delay):
        real_refresh(self)
        j1.t = time.monotonic() - delay      # J1 chưa trả lời lần hỏi này; lần trả lời gần nhất cách đây `delay` s

    monkeypatch.setattr(type(hw), "refresh_all", lambda self: refresh_without_j1(self, 0.2))
    r.poll()
    monkeypatch.setattr(type(hw), "refresh_all", lambda self: refresh_without_j1(self, 1.0))
    with pytest.raises(RobotFault):
        r.poll()


def test_enable_refuses_when_zero_is_wrong(robot):
    """Tay thả xuôi nhưng đọc J1 = 178° mà offset = 0 (zero motor sai, hoặc thiếu offset 180° của tay trái v1.0):
    không được bật motor."""
    from openarm_shadow.robot.openarm_can_robot import RobotFault
    r, hw = robot
    hw.a.ms[0].q = np.deg2rad(178.2)
    r.connect()
    assert r.out_of_range()
    with pytest.raises(RobotFault):
        r.enable()
    assert not hw.enabled


def test_software_offset_shifts_motor_limits(monkeypatch):
    """J4 thẳng tay đọc đúng offset của first_real.yaml (zero lệch): được bật motor, lệnh 0° URDF = offset motor."""
    oa = make_fake_openarm_can()
    monkeypatch.setitem(sys.modules, "openarm_can", oa)
    from openarm_shadow.robot.openarm_can_robot import OpenArmCANRobot
    cfg = load_config("config/first_real.yaml")
    cfg["robot"]["feedback_timeout_s"] = 1.0
    off = cfg["robot"]["urdf_to_motor"]["right"]["offset_deg"][3]
    assert off < 0
    r = OpenArmCANRobot(cfg["robot"], ["right"])
    hw = oa.OpenArm.instances[-1]
    hw.a.ms[3].q = np.deg2rad(off)
    q = r.connect()["right"]
    assert not r.out_of_range()
    assert q[3] == pytest.approx(0.0, abs=1e-6)
    r.enable()
    r.send({"right": np.zeros(8)})
    assert hw.a.ms[3].q == pytest.approx(np.deg2rad(off), abs=np.deg2rad(0.5))


def test_close_disables_motors_even_if_damping_step_fails(robot):
    """CAN lỗi giữa bước giảm chấn (vd bus-off): motor vẫn phải được tắt."""
    r, hw = robot
    r.connect()
    r.enable()
    assert hw.enabled

    def boom(ps):
        raise RuntimeError("bus-off")

    hw.a.mit_control_all = boom
    with pytest.raises(RuntimeError):
        r.close()
    assert not hw.enabled and not r.enabled


def left_robot(monkeypatch, j1_motor_deg):
    """Tay trái v1.0: motor J1/J2 lắp lệch 180° (openarm_driver joint_offsets = pi) -> offset phần mềm 180°."""
    oa = make_fake_openarm_can()
    monkeypatch.setitem(sys.modules, "openarm_can", oa)
    from openarm_shadow.robot.openarm_can_robot import OpenArmCANRobot
    cfg = load_config()
    cfg["robot"]["feedback_timeout_s"] = 1.0
    cfg["robot"]["urdf_to_motor"]["left"]["offset_deg"] = [180, 180, 0, 0, 0, 0, 0]
    cfg["robot"]["gripper"]["left"] = {"open_deg": -5.3, "closed_deg": 54.7}
    r = OpenArmCANRobot(cfg["robot"], ["left"])
    hw = oa.OpenArm.instances[-1]
    hw.a.ms[0].q, hw.a.ms[1].q = np.deg2rad(j1_motor_deg), np.deg2rad(180.2)
    return r, hw


@pytest.mark.parametrize("j1_motor_deg", [178.1, -181.9])     # cùng một tư thế, số đọc ở hai phía ±180°
def test_left_arm_180_offset_reads_near_zero_and_never_whips(monkeypatch, j1_motor_deg):
    r, hw = left_robot(monkeypatch, j1_motor_deg)
    q = r.connect()["left"]
    assert np.rad2deg(q[0]) == pytest.approx(-1.9, abs=0.01) and np.rad2deg(q[1]) == pytest.approx(0.2, abs=0.01)
    assert not r.out_of_range()
    r.enable()
    for _ in range(10):
        r.send({"left": np.append(np.deg2rad([10, 0, 0, 0, 0, 0, 0]), np.nan)})
    # Lệnh +10° URDF: motor đi +11.9° từ số đọc ban đầu, không quay vòng sang phía kia của ±180°
    assert np.rad2deg(hw.a.ms[0].q) == pytest.approx(j1_motor_deg + 11.9, abs=0.1)
    assert np.rad2deg(r.read()["left"][0]) == pytest.approx(10.0, abs=0.1)


def test_left_arm_garbage_still_rejected_with_offset(monkeypatch):
    r, hw = left_robot(monkeypatch, 178.1)
    r.connect()
    r.enable()
    hw.a.ms[0].glitch = [GARBAGE]
    r.send({"left": np.append(r.read()["left"][:7], np.nan)})
    assert np.rad2deg(r.arms["left"].q_motor[0]) == pytest.approx(178.1, abs=0.5)
    assert r.arms["left"].n_rejected >= 1


def test_left_arm_far_out_of_range_still_refused(monkeypatch):
    r, hw = left_robot(monkeypatch, 90.0)                     # URDF -90°: ngoài giới hạn J1 ±75°
    r.connect()
    assert r.out_of_range()
    from openarm_shadow.robot.openarm_can_robot import RobotFault
    with pytest.raises(RobotFault):
        r.enable()


def test_gripper_config_per_side(monkeypatch):
    r, _ = left_robot(monkeypatch, 178.1)
    a = r.arms["left"]
    assert np.rad2deg(a.grip_to_motor(0.0)) == pytest.approx(54.7) and np.rad2deg(a.grip_to_motor(1.0)) == \
        pytest.approx(-5.3)
    assert a.g_kp == 5.0                                      # khoá không ghi đè lấy từ robot.gripper chung
