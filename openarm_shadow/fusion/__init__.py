"""Hợp nhất nhiều camera thành một quan sát 3D: mô hình camera + hiệu chuẩn (camera_model), triangulation có trọng
số (triangulation), cùng một người ở mọi camera (person_match), bàn tay 3D (hand, hand_model, orientation),
điều phối + hợp nhất thân (multiview)."""
from .camera_model import CameraModel, check_image_size, load_calibration
from .hand import HandFuser
from .hand_model import HandShape, PalmModel, hand_forearm_angle
from .multiview import MultiViewPerception, view_options
from .orientation import OrientationFusion
from .person_match import OperatorMatcher
from .triangulation import REF_FX, err_conf_factor, fuse_point, reprojection_px, triangulate_weighted
