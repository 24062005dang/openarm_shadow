"""Ánh xạ tay người -> robot: retarget hướng chi kiểu SEW-Mimic (sew), độ mở kẹp (grip), và pipeline nối quan sát
-> hiệu chuẩn hướng tay -> retarget -> lọc khớp (pipeline)."""
from .grip import GripMapper
from .pipeline import ShadowPipeline
from .sew import (DEFAULT_HAND_NEUTRAL, ArmRetargeter, RetargetInfo, default_hand_neutral, mirror_rotation,
                  mirror_vector)
