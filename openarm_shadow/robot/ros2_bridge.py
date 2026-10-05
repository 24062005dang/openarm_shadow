"""Backend robot qua ROS 2: hệ thống này chỉ làm perception -> lệnh khớp; điều khiển motor là việc của backend OpenArm
bên kia (repo hoanglmv/openarm_can, nhánh hoang1: sim/openarm_joint_bridge.py + sim/dashboard_server.py).

Giao diện (docs/ROS2.md):
- nhận  /openarm/joint_states          sensor_msgs/JointState, 16 tên left_j1..left_j7, left_gripper, right_j1..
                                       right_gripper; khớp tay rad, kẹp m (hành trình ngón, 0 = đóng, 0.043 = mở hết)
- phát  /openarm/teleop/joint_commands  std_msgs/Float64MultiArray, ĐÚNG 14 số rad: left_j1..j7 rồi right_j1..j7
- phát  /openarm/teleop/left_gripper    std_msgs/Float64MultiArray, data[0] = hành trình kẹp (m)
- phát  /openarm/teleop/right_gripper   như trên

Quy ước góc: backend dùng URDF chính thức của OpenArm, góc khớp = góc motor thô (không offset, không đổi dấu).
Pipeline của ta cũng tính theo URDF chính thức nên gửi thẳng; ros2.urdf_to_ros (sign, offset) mặc định đồng nhất,
chỉ để chỉnh khi đo thấy lệch. Backend giữ mục tiêu cuối khi không có lệnh mới, và tự giới hạn tốc độ/kẹp giới hạn.

Hành vi:
- Chưa engage (hoặc --dry-run): KHÔNG phát gì; lệnh trong SafetyGate bám theo góc đo (runtime.controller), nên robot
  bị đẩy tay / bị backend khác điều khiển thì hình trong app đi theo, và lúc engage lệnh bắt đầu đúng tư thế thật.
- Đang engage: phát 14 khớp mỗi nhịp điều khiển (100 Hz). Tay không điều khiển (vd --arms right) gửi đúng góc đo
  hiện tại của nó, để backend không kéo tay đó đi đâu. Kẹp làm tròn về ros2.gripper.levels mức (GripLevels) rồi chỉ
  phát khi đổi >= min_change_m hoặc mỗi keepalive_s (backend in một dòng log và gửi lại khung POS_FORCE mỗi lệnh kẹp).
- joint_states cũ hơn state_timeout_s khi đang engage -> RobotFault: vòng điều khiển dừng, app thoát; backend giữ
  mục tiêu cuối.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from .openarm_can_robot import RobotFault

SIDES_ORDER = ("left", "right")      # thứ tự trong /openarm/teleop/joint_commands


def arm_names(side):
    return [f"{side}_j{i}" for i in range(1, 8)]


class JointMap:
    """Quy đổi giữa trạng thái của ta (mỗi tay 8 số: 7 góc URDF rad + độ mở kẹp 0..1) và topic của backend."""

    def __init__(self, rcfg: dict):
        m = rcfg.get("urdf_to_ros") or {}
        self.sign = {s: np.asarray((m.get(s) or {}).get("sign", [1] * 7), float) for s in SIDES_ORDER}
        self.offset = {s: np.deg2rad(np.asarray((m.get(s) or {}).get("offset_deg", [0] * 7), float))
                       for s in SIDES_ORDER}
        self.stroke_max = float((rcfg.get("gripper") or {}).get("stroke_max_m", 0.043))

    def to_ros(self, side, q_urdf):
        return self.sign[side] * np.asarray(q_urdf, float) + self.offset[side]

    def to_urdf(self, side, q_ros):
        return (np.asarray(q_ros, float) - self.offset[side]) * self.sign[side]

    def grip_to_stroke(self, f):
        return float(np.clip(f, 0.0, 1.0)) * self.stroke_max

    def stroke_to_grip(self, stroke):
        return float(np.clip(stroke / self.stroke_max, 0.0, 1.0)) if self.stroke_max > 0 else 0.0

    def parse_state(self, names, positions):
        """JointState (tên, vị trí) -> ({tay: 8 số theo quy ước của ta}, {tay: 7 góc thô của backend}).
        Thiếu khớp nào của một tay thì tay đó không có trong kết quả."""
        pos = {n: float(p) for n, p in zip(names, positions)}
        state, raw = {}, {}
        for s in SIDES_ORDER:
            keys = arm_names(s)
            if not all(k in pos for k in keys):
                continue
            raw[s] = np.array([pos[k] for k in keys])
            grip = pos.get(f"{s}_gripper")
            state[s] = np.append(self.to_urdf(s, raw[s]), np.nan if grip is None else self.stroke_to_grip(grip))
        return state, raw

    def arm_command(self, cmd, raw_meas):
        """cmd: {tay ta điều khiển: 8 số}. raw_meas: {tay: 7 góc thô đo được}. -> list 14 số hoặc None.
        Tay không điều khiển: gửi đúng góc đo (giữ yên); thiếu số đo của tay đó thì không gửi được."""
        out = []
        for s in SIDES_ORDER:
            if s in cmd:
                q = self.to_ros(s, np.asarray(cmd[s], float)[:7])
            elif s in raw_meas:
                q = raw_meas[s]
            else:
                return None
            if not np.all(np.isfinite(q)):
                return None
            out += [float(v) for v in q]
        return out


class GripLevels:
    """Làm tròn độ mở kẹp 0..1 về `levels` mức đều nhau (vd 10 mức: 0, 1/9, ..., 1 -> 0, 4,8 mm, ..., 43 mm), ngay
    trước khi phát, để backend chỉ nhận đúng các mức này. Có trễ: chỉ đổi mức khi giá trị vượt quá nửa bước +
    hysteresis (tính theo phần của một bước) khỏi mức hiện tại, tránh kẹp nhảy qua lại khi tay ở gần ranh giới.
    levels None / < 2: không làm tròn (liên tục)."""

    def __init__(self, levels=None, hysteresis=0.3):
        self.n = int(levels) if levels else 0
        self.step = 1.0 / (self.n - 1) if self.n >= 2 else 0.0
        self.hyst = float(hysteresis)
        self.idx = None

    def __call__(self, f):
        f = float(np.clip(f, 0.0, 1.0))
        if self.n < 2:
            return f
        if self.idx is None or abs(f - self.idx * self.step) > (0.5 + self.hyst) * self.step:
            self.idx = int(round(f / self.step))
        return self.idx * self.step


class Ros2Robot:
    """Cùng giao diện SimRobot / OpenArmCANRobot (connect, read, poll, enable, send, close).

    cfg: config đầy đủ (dùng ros2 và safety.soft_limits_deg). sides: các tay ta điều khiển."""

    follows_measured = True          # runtime.controller: chưa engage thì lệnh bám theo góc đo

    def __init__(self, cfg: dict, sides):
        self.rcfg = dict(cfg.get("ros2") or {})
        self.sides = list(sides)
        self.map = JointMap(self.rcfg)
        lim = cfg.get("safety", {}).get("soft_limits_deg", {})
        self.limits = {s: np.deg2rad(np.asarray(lim[s], float)) for s in self.sides if s in lim}
        self.limit_tol = np.deg2rad(float(self.rcfg.get("limit_tol_deg", 5.0)))
        self.state_timeout = float(self.rcfg.get("state_timeout_s", 0.5))
        gc = self.rcfg.get("gripper") or {}
        self.grip_on = bool(gc.get("enabled", True))
        self.grip_min_change = float(gc.get("min_change_m", 0.0005))
        self.grip_keepalive = float(gc.get("keepalive_s", 0.5))
        self.grip_levels = {s: GripLevels(gc.get("levels"), gc.get("level_hysteresis", 0.3)) for s in SIDES_ORDER}
        self.lock = threading.Lock()
        self.state, self.raw, self.t_state = {}, {}, None
        self.n_states = 0
        self.enabled = False
        self.engaged = False
        self.n_sent = 0
        self._grip_last = {s: (None, -np.inf) for s in SIDES_ORDER}
        self.node = self.executor = self.thread = None
        self._own_rclpy = False

    # -- ROS ------------------------------------------------------------------------------------------------
    def _on_state(self, msg):
        state, raw = self.map.parse_state(msg.name, msg.position)
        if not state:
            return
        with self.lock:
            self.state.update(state)
            self.raw.update(raw)
            self.t_state = time.monotonic()
            self.n_states += 1

    def _start_ros(self):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray
        if not rclpy.ok():
            rclpy.init()
            self._own_rclpy = True
        rc = self.rcfg
        self.node = rclpy.create_node(rc.get("node_name", "openarm_shadow_teleop"))
        # BEST_EFFORT nhận được từ publisher RELIABLE (joint_states) và khớp subscriber BEST_EFFORT của bridge.
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.node.create_subscription(JointState, rc.get("joint_states_topic", "/openarm/joint_states"),
                                      self._on_state, qos)
        self._Msg = Float64MultiArray
        self.pub_arm = self.node.create_publisher(
            Float64MultiArray, rc.get("arm_command_topic", "/openarm/teleop/joint_commands"), qos)
        self.pub_grip = {
            "left": self.node.create_publisher(
                Float64MultiArray, rc.get("left_gripper_topic", "/openarm/teleop/left_gripper"), qos),
            "right": self.node.create_publisher(
                Float64MultiArray, rc.get("right_gripper_topic", "/openarm/teleop/right_gripper"), qos)}
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True, name="ros2")
        self.thread.start()

    def _publish(self, pub, data):
        msg = self._Msg()
        msg.data = [float(v) for v in data]
        pub.publish(msg)

    # -- giao diện robot ----------------------------------------------------------------------------------------
    def connect(self):
        """Khởi động node, chờ joint_states có đủ các tay ta điều khiển. -> trạng thái (góc URDF + kẹp 0..1)."""
        self._start_ros()
        timeout = float(self.rcfg.get("connect_timeout_s", 5.0))
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            with self.lock:
                if all(s in self.state for s in self.sides):
                    break
            time.sleep(0.02)
        else:
            topic = self.rcfg.get("joint_states_topic", "/openarm/joint_states")
            raise RobotFault(f"Không nhận được {topic} đủ khớp tay {self.sides} sau {timeout:.0f} s. Backend OpenArm "
                             "đã chạy chưa, cùng ROS_DOMAIN_ID chưa? Thử: ros2 topic echo --once " + topic)
        bad = self.out_of_range()
        if bad:
            print("CẢNH BÁO: góc đo nằm ngoài giới hạn mềm -> quy ước góc / zero của robot khác URDF? "
                  "KHÔNG gửi lệnh:")
            for line in bad:
                print("   ", line)
        return self.read()

    def out_of_range(self):
        """Khớp có góc đo (đã quy về URDF) ngoài safety.soft_limits_deg (± limit_tol_deg)."""
        out = []
        st = self.read()
        for s, lim in self.limits.items():
            q = st[s][:7]
            for i in np.flatnonzero((q < lim[:, 0] - self.limit_tol) | (q > lim[:, 1] + self.limit_tol)):
                out.append(f"{s} J{i + 1}: {np.rad2deg(q[i]):7.1f}° (giới hạn {np.rad2deg(lim[i, 0]):.0f}.."
                           f"{np.rad2deg(lim[i, 1]):.0f}°)")
        return out

    def read(self):
        with self.lock:
            return {s: self.state[s].copy() for s in self.sides if s in self.state}

    def age(self):
        with self.lock:
            return np.inf if self.t_state is None else time.monotonic() - self.t_state

    def poll(self):
        """--dry-run: đọc góc mới nhất (không gửi gì)."""
        if self.age() > max(self.state_timeout, 1.0):
            raise RobotFault(f"joint_states không cập nhật {self.age():.1f} s")
        return self.read()

    def enable(self):
        """Không bật motor (việc của backend): chỉ cho phép gửi lệnh. Từ chối nếu góc đo ngoài giới hạn mềm."""
        bad = self.out_of_range()
        if bad:
            raise RobotFault("Không gửi lệnh: góc đo ngoài giới hạn mềm (" + "; ".join(bad) + "). Kiểm tra quy ước "
                             "góc của backend (ros2.urdf_to_ros) và zero của robot, xem docs/ROS2.md.")
        self.enabled = True

    def set_engaged(self, engaged):
        self.engaged = bool(engaged)

    def send(self, cmd, dq=None):
        """Phát lệnh khi đã enable VÀ đang engage. dq (feedforward vận tốc) không có trên giao diện này: bỏ qua."""
        if not (self.enabled and self.engaged):
            return
        if self.age() > self.state_timeout:
            raise RobotFault(f"joint_states không cập nhật {self.age():.2f} s (> {self.state_timeout} s): dừng gửi lệnh")
        with self.lock:
            raw = {s: v.copy() for s, v in self.raw.items()}
        arm = self.map.arm_command({s: cmd[s] for s in self.sides if s in cmd}, raw)
        if arm is None:
            return
        self._publish(self.pub_arm, arm)
        self.n_sent += 1
        if not self.grip_on:
            return
        now = time.monotonic()
        for s in self.sides:
            f = np.asarray(cmd[s], float)[7] if s in cmd else np.nan
            if not np.isfinite(f):
                continue
            stroke = self.map.grip_to_stroke(self.grip_levels[s](f))
            last, t_last = self._grip_last[s]
            if last is None or abs(stroke - last) >= self.grip_min_change or now - t_last >= self.grip_keepalive:
                self._publish(self.pub_grip[s], [stroke])
                self._grip_last[s] = (stroke, now)

    def relax(self):
        pass

    def close(self):
        """Dừng node. Không gửi gì thêm: backend giữ mục tiêu cuối."""
        self.enabled = self.engaged = False
        if self.executor is not None:
            self.executor.shutdown(timeout_sec=1.0)
        if self.thread is not None:
            self.thread.join(timeout=2.0)      # chờ luồng spin thoát hẳn, không thì rclpy abort lúc tắt chương trình
        if self.node is not None:
            self.node.destroy_node()
        if self._own_rclpy:
            import rclpy
            if rclpy.ok():
                rclpy.shutdown()
        if self.n_sent:
            print(f"ROS 2: đã gửi {self.n_sent} lệnh, nhận {self.n_states} joint_states")
