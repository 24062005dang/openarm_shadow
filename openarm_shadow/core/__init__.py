"""Toán dùng chung: hình học + bài toán con SP1/SP2/SP4 (geometry), phép quay (rotations), động học OpenArm v1.0
từ URDF (kinematics). Không phụ thuộc camera, MediaPipe hay robot."""
from .geometry import angle_between, make_frame, orthonormalize, rot, seg_seg_distance, sp1, sp2, sp4, unit, wrap
from .kinematics import ArmKinematics
from .rotations import matrix_to_quaternion, quaternion_to_matrix, rotation_distance, slerp_rotation
