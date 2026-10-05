"""Node ROS 2 `teleop`: chạy toàn bộ openarm_shadow (camera -> perception -> retarget -> SafetyGate) với backend
--robot ros2: nhận /openarm/joint_states, phát /openarm/teleop/joint_commands + left_gripper + right_gripper.

    ros2 run openarm_shadow_ros teleop --source multi --config /duong/dan/local_3cam.yaml [--dry-run] [--arms right]

Tham số giống scripts/shadow.py (mặc định --robot ros2). Thư viện openarm_shadow phải import được: cài bằng
`pip install -e <repo>` hoặc đặt biến môi trường OPENARM_SHADOW_ROOT=<repo>. Xem docs/ROS2.md.
"""
import os
import sys


def _import_cli():
    try:
        from openarm_shadow.runtime import cli
    except ImportError:
        root = os.environ.get("OPENARM_SHADOW_ROOT")
        if not root:
            raise SystemExit("Không import được openarm_shadow. Cài: pip install -e <repo Openarm_chaylai> hoặc đặt "
                             "OPENARM_SHADOW_ROOT=<repo Openarm_chaylai>")
        sys.path.insert(0, root)
        from openarm_shadow.runtime import cli
    return cli


def main(argv=None):
    from rclpy.utilities import remove_ros_args
    args = remove_ros_args(sys.argv if argv is None else argv)[1:]
    _import_cli().main(args, default_robot="ros2")


if __name__ == "__main__":
    main()
