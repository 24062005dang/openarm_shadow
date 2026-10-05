"""Backend robot: SimRobot (không cần phần cứng), OpenArmCAN (openarm_can, CAN-FD) và Ros2Robot (gửi lệnh khớp cho
backend OpenArm qua ROS 2, xem docs/ROS2.md)."""
from .sim import SimRobot


REAL_KINDS = ("openarm", "ros2")     # robot thật: hỏi xác nhận, tự engage theo auto_engage_real_s


def make_robot(kind: str, cfg: dict, sides):
    if kind == "sim":
        return SimRobot(sides)
    if kind == "openarm":
        from .openarm_can_robot import OpenArmCANRobot
        return OpenArmCANRobot(cfg["robot"], sides)
    if kind == "ros2":
        from .ros2_bridge import Ros2Robot
        return Ros2Robot(cfg, sides)
    raise ValueError(f"robot không hỗ trợ: {kind}")
