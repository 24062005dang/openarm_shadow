"""Backend robot: SimRobot (không cần phần cứng) và OpenArmCAN (openarm_can, CAN-FD)."""
from .sim import SimRobot


def make_robot(kind: str, cfg: dict, sides):
    if kind == "sim":
        return SimRobot(sides)
    if kind == "openarm":
        from .openarm_can_robot import OpenArmCANRobot
        return OpenArmCANRobot(cfg["robot"], sides)
    raise ValueError(f"robot không hỗ trợ: {kind}")
