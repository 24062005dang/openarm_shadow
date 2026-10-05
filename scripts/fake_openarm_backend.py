#!/usr/bin/env python3
"""Backend OpenArm GIẢ để thử node teleop qua ROS 2 khi không có robot (mô phỏng repo hoanglmv/openarm_can nhánh
hoang1: sim/openarm_joint_bridge.py + bộ nội suy của sim/dashboard_server.py). Cần ROS 2 (rclpy).

- Phát /openarm/joint_states (sensor_msgs/JointState, RELIABLE depth 5) 100 Hz, 16 tên, rad / m.
- Nhận /openarm/teleop/joint_commands (Float64MultiArray, đúng 14 số; sai số lượng thì bỏ như bridge thật) và
  /openarm/teleop/left_gripper, right_gripper (data[0], m). Mục tiêu kẹp theo JOINT_LIMITS, giữ tới lệnh kế tiếp.
- Tư thế tiến tới mục tiêu tối đa --velocity-limit rad/s (backend thật mặc định 0,25), kẹp 2,5 rad/s ~ nhanh.
- --push: từ giây 1 tới 3, nếu chưa nhận lệnh nào, "đẩy tay" left_j4 lên 0,6 rad (thử đồng bộ khi chưa engage).
- Ghi MỌI lệnh nhận được + trạng thái ra --out (.npz) để phân tích bằng scripts/analyze_ros2_log.py.
- Thoát khi hết --duration, hoặc --idle-exit giây sau lệnh cuối cùng (khi đã từng nhận lệnh).

    python3 scripts/fake_openarm_backend.py --out backend.npz --duration 300 --push
"""
import argparse
import math
import time

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

NAMES = ["left_j1", "left_j2", "left_j3", "left_j4", "left_j5", "left_j6", "left_j7", "left_gripper",
         "right_j1", "right_j2", "right_j3", "right_j4", "right_j5", "right_j6", "right_j7", "right_gripper"]
# JOINT_LIMITS nhánh hoang1 (sim/config.py), theo thứ tự NAMES
LIMITS = np.array([(-3.4907, 1.3963), (-3.3161, 0.17453), (-1.5708, 1.5708), (0.0, 2.4435), (-1.5708, 1.5708),
                   (-0.7854, 0.7854), (-1.5708, 1.5708), (0.0, 0.043),
                   (-1.3963, 3.4907), (-0.17453, 3.3161), (-1.5708, 1.5708), (0.0, 2.4435), (-1.5708, 1.5708),
                   (-0.7854, 0.7854), (-1.5708, 1.5708), (0.0, 0.043)])
ARM_IDX = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, default=600.0)
    ap.add_argument("--idle-exit", type=float, default=3.0)
    ap.add_argument("--velocity-limit", type=float, default=0.25)
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    rclpy.init()
    node = rclpy.create_node("fake_openarm_backend")
    q = np.zeros(16)
    q[7] = q[15] = 0.02                         # kẹp mở 20 mm
    target = q.copy()
    t0 = time.monotonic()
    log = {"arm_t": [], "arm": [], "grip_t": [], "grip_side": [], "grip": [], "state_t": [], "state": [],
           "rejected": 0}
    last_cmd = [None]

    def on_arm(msg):
        d = list(msg.data)
        if len(d) != 14:                         # bridge thật: "Rejected teleop arm command"
            log["rejected"] += 1
            return
        now = time.monotonic() - t0
        log["arm_t"].append(now)
        log["arm"].append(d)
        last_cmd[0] = now
        for k, i in enumerate(ARM_IDX):
            if math.isfinite(d[k]):
                target[i] = min(LIMITS[i, 1], max(LIMITS[i, 0], d[k]))

    def on_grip(side, i):
        def cb(msg):
            if not msg.data:
                return
            now = time.monotonic() - t0
            log["grip_t"].append(now)
            log["grip_side"].append(0 if side == "left" else 1)
            log["grip"].append(float(msg.data[0]))
            last_cmd[0] = now
            target[i] = min(0.043, max(0.0, float(msg.data[0])))
        return cb

    sub_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
    node.create_subscription(Float64MultiArray, "/openarm/teleop/joint_commands", on_arm, sub_qos)
    node.create_subscription(Float64MultiArray, "/openarm/teleop/left_gripper", on_grip("left", 7), sub_qos)
    node.create_subscription(Float64MultiArray, "/openarm/teleop/right_gripper", on_grip("right", 15), sub_qos)
    pub = node.create_publisher(JointState, "/openarm/joint_states",
                                QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=5,
                                           reliability=ReliabilityPolicy.RELIABLE))
    dt = 0.01
    vmax = np.full(16, args.velocity_limit)
    vmax[[7, 15]] = 2.5 * 0.043 / 1.20          # kẹp 2,5 rad/s quy ra m/s

    def tick():
        now = time.monotonic() - t0
        if args.push and last_cmd[0] is None and 1.0 <= now <= 3.0:
            target[3] = q[3] = 0.3 * (now - 1.0)
        q[:] = q + np.clip(target - q, -vmax * dt, vmax * dt)
        msg = JointState()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.name, msg.position = NAMES, [float(v) for v in q]
        msg.velocity, msg.effort = [0.0] * 16, [0.0] * 16
        pub.publish(msg)
        log["state_t"].append(now)
        log["state"].append(q.copy())

    node.create_timer(dt, tick)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    print(f"[fake backend] chạy tối đa {args.duration:.0f} s, ghi {args.out}", flush=True)
    try:
        while time.monotonic() - t0 < args.duration:
            ex.spin_once(timeout_sec=0.05)
            if last_cmd[0] is not None and time.monotonic() - t0 - last_cmd[0] > args.idle_exit:
                break
    except KeyboardInterrupt:
        pass
    np.savez(args.out, arm_t=np.array(log["arm_t"]), arm=np.array(log["arm"]).reshape(-1, 14),
             grip_t=np.array(log["grip_t"]), grip_side=np.array(log["grip_side"], int), grip=np.array(log["grip"]),
             state_t=np.array(log["state_t"]), state=np.array(log["state"]).reshape(-1, 16),
             rejected=log["rejected"], velocity_limit=args.velocity_limit, names=np.array(NAMES))
    print(f"[fake backend] nhận {len(log['arm_t'])} lệnh khớp, {len(log['grip_t'])} lệnh kẹp, "
          f"bỏ {log['rejected']} lệnh sai số lượng -> {args.out}", flush=True)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
