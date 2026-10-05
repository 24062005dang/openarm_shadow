"""Backend ROS 2 (robot.ros2_bridge) với rclpy giả: quy đổi topic, chỉ phát khi engage, đồng bộ với robot thật."""
import sys
import threading
import time
import types

import numpy as np
import pytest

from openarm_shadow.config import load_config
from openarm_shadow.core.kinematics import ArmKinematics
from openarm_shadow.robot.openarm_can_robot import RobotFault
from openarm_shadow.robot.ros2_bridge import JointMap, Ros2Robot
from openarm_shadow.runtime.controller import Controller
from openarm_shadow.safety import SafetyGate

NAMES = [f"{s}_j{i}" for s in ("left",) for i in range(1, 8)] + ["left_gripper"] + \
        [f"{s}_j{i}" for s in ("right",) for i in range(1, 8)] + ["right_gripper"]


class FakeNode:
    def __init__(self, name):
        self.name, self.subs, self.pubs = name, {}, {}

    def create_subscription(self, msg_type, topic, cb, qos):
        self.subs[topic] = cb

    def create_publisher(self, msg_type, topic, qos):
        pub = types.SimpleNamespace(topic=topic, sent=[])
        pub.publish = lambda msg, p=pub: p.sent.append(list(msg.data))
        self.pubs[topic] = pub
        return pub

    def destroy_node(self):
        pass


class FakeExecutor:
    def __init__(self):
        self.stop = threading.Event()

    def add_node(self, node):
        pass

    def spin(self):
        self.stop.wait()

    def shutdown(self, timeout_sec=None):
        self.stop.set()


@pytest.fixture
def fake_ros(monkeypatch):
    state = {"ok": False, "nodes": []}
    rclpy = types.ModuleType("rclpy")
    rclpy.ok = lambda: state["ok"]
    rclpy.init = lambda: state.update(ok=True)
    rclpy.shutdown = lambda: state.update(ok=False)

    def create_node(name):
        n = FakeNode(name)
        state["nodes"].append(n)
        return n
    rclpy.create_node = create_node
    ex = types.ModuleType("rclpy.executors")
    ex.SingleThreadedExecutor = FakeExecutor
    qos = types.ModuleType("rclpy.qos")
    qos.QoSProfile = lambda **kw: kw
    qos.HistoryPolicy = types.SimpleNamespace(KEEP_LAST="keep_last")
    qos.ReliabilityPolicy = types.SimpleNamespace(BEST_EFFORT="best_effort")
    sensor = types.ModuleType("sensor_msgs.msg")
    sensor.JointState = object
    std = types.ModuleType("std_msgs.msg")
    std.Float64MultiArray = lambda: types.SimpleNamespace(data=[])
    for name, mod in {"rclpy": rclpy, "rclpy.executors": ex, "rclpy.qos": qos, "sensor_msgs": types.ModuleType("s"),
                      "sensor_msgs.msg": sensor, "std_msgs": types.ModuleType("m"), "std_msgs.msg": std}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return state


def joint_state(q=None):
    """Thông điệp joint_states: q = {tên: giá trị}, mặc định tất cả 0, kẹp mở 20 mm."""
    pos = {n: (0.02 if n.endswith("gripper") else 0.0) for n in NAMES}
    pos.update(q or {})
    return types.SimpleNamespace(name=list(NAMES), position=[pos[n] for n in NAMES])


def make_robot(fake_ros, sides=("right", "left"), cfg_extra=None):
    cfg = load_config()
    cfg["ros2"]["connect_timeout_s"] = 1.0
    if cfg_extra:
        cfg["ros2"].update(cfg_extra)
    r = Ros2Robot(cfg, sides)
    t = threading.Thread(target=lambda: (time.sleep(0.05), feed(fake_ros)))
    t.start()
    q = r.connect()
    t.join()
    return r, q


def node(fake_ros):
    return fake_ros["nodes"][-1]


def feed(fake_ros, q=None):
    node(fake_ros).subs["/openarm/joint_states"](joint_state(q))


def sent(fake_ros, topic):
    return node(fake_ros).pubs[topic].sent


# ---------------------------------------------------------------------------------------------------------------
def test_joint_map_order_gripper_and_hold_other_arm():
    m = JointMap({"gripper": {"stroke_max_m": 0.043}})
    state, raw = m.parse_state(NAMES, joint_state({"left_j1": -0.4, "right_j4": 1.2, "right_gripper": 0.043}).position)
    assert state["left"][0] == pytest.approx(-0.4) and state["right"][3] == pytest.approx(1.2)
    assert state["right"][7] == pytest.approx(1.0) and state["left"][7] == pytest.approx(0.02 / 0.043)
    cmd = {"right": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.5])}
    arm = m.arm_command(cmd, raw)
    assert len(arm) == 14
    assert arm[:7] == pytest.approx(raw["left"])            # tay trái không điều khiển: giữ đúng góc đo
    assert arm[7:] == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    assert m.grip_to_stroke(0.5) == pytest.approx(0.0215) and m.grip_to_stroke(2.0) == pytest.approx(0.043)
    assert m.arm_command(cmd, {}) is None                     # thiếu số đo tay trái -> không gửi được


def test_joint_map_sign_offset_roundtrip():
    m = JointMap({"urdf_to_ros": {"left": {"sign": [-1, 1, 1, 1, 1, 1, 1], "offset_deg": [180, 0, 0, 0, 0, 0, 0]}}})
    q = np.array([0.3, -0.2, 0.1, 0.5, 0, 0, 0])
    ros = m.to_ros("left", q)
    assert ros[0] == pytest.approx(np.pi - 0.3)
    assert m.to_urdf("left", ros) == pytest.approx(q)
    assert m.to_ros("right", q) == pytest.approx(q)


def test_publishes_only_when_enabled_and_engaged(fake_ros):
    r, q = make_robot(fake_ros, sides=("right",))
    assert set(q) == {"right"}
    cmd = {"right": np.array([0.1, 0, 0, 0.5, 0, 0, 0, 1.0])}
    r.send(cmd)                                               # chưa enable
    r.enable()
    r.send(cmd)                                               # chưa engage
    assert sent(fake_ros, "/openarm/teleop/joint_commands") == []
    r.set_engaged(True)
    r.send(cmd)
    arm = sent(fake_ros, "/openarm/teleop/joint_commands")
    assert len(arm) == 1 and len(arm[0]) == 14 and arm[0][7] == pytest.approx(0.1) and arm[0][10] == pytest.approx(0.5)
    assert sent(fake_ros, "/openarm/teleop/right_gripper") == [[pytest.approx(0.043)]]
    assert sent(fake_ros, "/openarm/teleop/left_gripper") == []      # không điều khiển tay trái
    r.close()


def test_gripper_sent_on_change_or_keepalive(fake_ros):
    r, _ = make_robot(fake_ros, cfg_extra={"gripper": {"enabled": True, "min_change_m": 0.001, "keepalive_s": 0.2}})
    r.enable()
    r.set_engaged(True)
    base = {s: np.zeros(8) for s in ("right", "left")}
    for f in (0.5, 0.5 + 1e-3, 0.5 + 2e-3, 0.6):              # 0,04 mm / 0,09 mm: chưa đủ 1 mm
        cmd = {s: v.copy() for s, v in base.items()}
        cmd["right"][7] = f
        cmd["left"][7] = np.nan                               # kẹp trái đang giữ: không gửi
        feed(fake_ros)
        r.send(cmd)
    g = sent(fake_ros, "/openarm/teleop/right_gripper")
    assert [round(v[0], 5) for v in g] == [round(0.5 * 0.043, 5), round(0.6 * 0.043, 5)]
    assert sent(fake_ros, "/openarm/teleop/left_gripper") == []
    time.sleep(0.25)
    feed(fake_ros)
    r.send(cmd)
    assert len(sent(fake_ros, "/openarm/teleop/right_gripper")) == 3          # keepalive
    assert len(sent(fake_ros, "/openarm/teleop/joint_commands")) == 5
    r.close()


def test_connect_timeout_and_out_of_range(fake_ros):
    cfg = load_config()
    cfg["ros2"]["connect_timeout_s"] = 0.2
    with pytest.raises(RobotFault):
        Ros2Robot(cfg, ["right"]).connect()                  # không có joint_states
    # Tay trái thả xuôi mà J1 đọc 178° (zero/quy ước khác URDF): không cho gửi lệnh
    r, _ = make_robot(fake_ros)
    feed(fake_ros, {"left_j1": np.deg2rad(178.0)})
    assert any("left J1" in line for line in r.out_of_range())
    with pytest.raises(RobotFault):
        r.enable()
    feed(fake_ros)
    r.enable()
    r.close()


def test_stale_joint_states_stops_sending(fake_ros):
    r, _ = make_robot(fake_ros, cfg_extra={"state_timeout_s": 0.1})
    r.enable()
    r.set_engaged(True)
    time.sleep(0.15)
    with pytest.raises(RobotFault):
        r.send({s: np.zeros(8) for s in ("right", "left")})
    r.close()


def test_controller_follows_measured_until_engaged(fake_ros):
    """Robot bị đẩy tay khi chưa engage: lệnh bám theo góc đo, không phát gì; engage xong mới phát, bắt đầu từ
    đúng tư thế thật (không kéo về tư thế cũ)."""
    r, q0 = make_robot(fake_ros)
    r.enable()
    cfg = load_config()
    kins = {s: ArmKinematics(s) for s in ("right", "left")}
    gate = SafetyGate(kins, cfg["safety"])
    gate.reset(q0)
    ctl = Controller(r, gate, 100)
    ctl.start()
    try:
        for k in range(4):                                     # đẩy khuỷu phải tới 40°
            feed(fake_ros, {"right_j4": np.deg2rad(10 * (k + 1))})
            time.sleep(0.03)
        time.sleep(0.05)
        assert sent(fake_ros, "/openarm/teleop/joint_commands") == []
        with ctl.lock:
            assert gate.cmd["right"][3] == pytest.approx(np.deg2rad(40))
            gate.set_target({s: np.append(gate.cmd[s][:7], np.nan) for s in gate.sides}, time.monotonic())
            gate.engage(time.monotonic())
        for _ in range(10):
            feed(fake_ros, {"right_j4": np.deg2rad(40)})
            time.sleep(0.01)
        arm = sent(fake_ros, "/openarm/teleop/joint_commands")
        assert arm and all(a[10] == pytest.approx(np.deg2rad(40), abs=1e-6) for a in arm)
        with ctl.lock:
            gate.disengage()
        time.sleep(0.05)
        n = len(sent(fake_ros, "/openarm/teleop/joint_commands"))
        time.sleep(0.05)
        assert len(sent(fake_ros, "/openarm/teleop/joint_commands")) == n     # nhả: ngừng phát
    finally:
        ctl.running = False
        ctl.join(1.0)
        r.close()
    assert ctl.error is None


def test_grip_levels_quantize_with_hysteresis():
    from openarm_shadow.robot.ros2_bridge import GripLevels
    g = GripLevels(10, 0.3)
    step = 1 / 9
    allowed = [k * step for k in range(10)]
    assert g(0.0) == 0.0 and g(1.0) == pytest.approx(1.0)
    assert g(0.52) == pytest.approx(allowed[5])
    assert g(allowed[5] + 0.7 * step) == pytest.approx(allowed[5])  # chưa vượt (0,5 + 0,3) bước: giữ mức
    assert g(allowed[5] + 0.85 * step) == pytest.approx(allowed[6])  # vượt: lên mức mới
    assert g(allowed[6] - 0.7 * step) == pytest.approx(allowed[6])  # đi xuống cũng có trễ
    vals = {round(g(f), 9) for f in np.random.default_rng(0).random(500)}
    assert vals <= {round(v, 9) for v in allowed}
    assert GripLevels(None)(0.37) == pytest.approx(0.37)           # không đặt mức: liên tục


def test_published_gripper_uses_levels(fake_ros):
    r, _ = make_robot(fake_ros, sides=("right",),
                      cfg_extra={"gripper": {"enabled": True, "levels": 10, "min_change_m": 0.0005}})
    r.enable()
    r.set_engaged(True)
    for f in np.linspace(0, 1, 40):                                   # kẹp mở dần từ 0 tới 1
        feed(fake_ros)
        r.send({"right": np.append(np.zeros(7), f)})
    g = [v[0] for v in sent(fake_ros, "/openarm/teleop/right_gripper")]
    assert len(g) == 10                                                # đúng 10 mức, mỗi mức gửi 1 lần
    assert g == pytest.approx([k * 0.043 / 9 for k in range(10)])
    r.close()
