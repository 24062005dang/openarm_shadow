"""Cấu hình chung cho một tay OpenArm v1.0 qua openarm_can.

Motor: J1-J2 DM8009, J3-J4 DM4340, J5-J7 DM4310, gripper DM4310
(v1.0 và v2 dùng cùng bộ motor, xem openarm-can-zero-position-calibration).
CAN ID gửi 0x01-0x08, nhận 0x11-0x18.
Gain lấy từ openarm_description/assets/robot/openarm_v1.0/config/arm/control_gains.yaml.
Gripper v1.0: hành trình motor khoảng [-60°, 0°] (MECH_LIM_V1).
"""
import math

import openarm_can as oa

ARM_TYPES = [oa.MotorType.DM8009, oa.MotorType.DM8009,
             oa.MotorType.DM4340, oa.MotorType.DM4340,
             oa.MotorType.DM4310, oa.MotorType.DM4310, oa.MotorType.DM4310]
ARM_SEND = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07]
ARM_RECV = [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17]
GRIP_TYPE, GRIP_SEND, GRIP_RECV = oa.MotorType.DM4310, 0x08, 0x18

# Gain MIT chính thức v1.0 (openarm_description .../openarm_v1.0/config/arm/control_gains.yaml)
KP = [70.0, 70.0, 70.0, 60.0, 10.0, 10.0, 10.0]
KD = [2.75, 2.5, 2.0, 2.0, 0.7, 0.6, 0.5]

# Giới hạn cơ khí ở mức motor (độ), từ openarm-can-zero-position-calibration
MECH_LIM_DEG = [(-80, 200), (-100, 100), (-90, 90), (0, 140),
                (-90, 90), (-45, 45), (-90, 90)]

NAMES = ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "Grip"]


def make_openarm(iface, with_gripper=True):
    arm = oa.OpenArm(iface, True)  # True = CAN-FD
    arm.init_arm_motors(ARM_TYPES, ARM_SEND, ARM_RECV)
    if with_gripper:
        arm.init_gripper_motor(GRIP_TYPE, GRIP_SEND, GRIP_RECV)
    arm.set_callback_mode_all(oa.CallbackMode.STATE)
    return arm


def read_positions(arm, with_gripper=True):
    """Gửi lệnh refresh (không tạo mô-men) rồi đọc vị trí (rad)."""
    arm.refresh_all()
    arm.recv_all(2000)
    q = [m.get_position() for m in arm.get_arm().get_motors()]
    if with_gripper:
        q += [m.get_position() for m in arm.get_gripper().get_motors()]
    return q


def link_ok(arm, n=7, max_age_s=0.2):
    """True/False theo từng khớp; None nếu bản openarm_can không có get_link_stats."""
    comp = arm.get_arm()
    if not hasattr(comp, "get_link_stats"):
        return None
    return [comp.get_link_stats(i).seconds_since_response() < max_age_s for i in range(n)]


def fmt_deg(q):
    return "  ".join(
        f"{name}:{'  ---  ' if not math.isfinite(v) else f'{math.degrees(v):7.1f}'}"
        for name, v in zip(NAMES, q))
