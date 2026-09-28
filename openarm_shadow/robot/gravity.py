"""Bù trọng lực tuỳ chọn bằng Pinocchio (pip install pin) từ URDF v1.0 đầy đủ (có khối lượng).

CHƯA KIỂM CHỨNG TRÊN ROBOT. Bật trong config (robot.gravity_comp.enabled) chỉ sau khi đã thử
từng khớp với gain thấp. URDF: openarm_description/assets/robot/openarm_v1.0/urdf/example/v1.urdf
"""
import numpy as np


class GravityModel:
    def __init__(self, urdf_path, sides):
        import pinocchio as pin
        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        self.idx = {}
        for s in sides:
            ids = []
            for i in range(1, 8):
                jid = self.model.getJointId(f"openarm_{s}_joint{i}")
                if jid >= self.model.njoints:
                    raise ValueError(f"URDF không có khớp openarm_{s}_joint{i}")
                ids.append((self.model.joints[jid].idx_q, self.model.joints[jid].idx_v))
            self.idx[s] = ids

    def torques(self, q_urdf: dict):
        q = self.pin.neutral(self.model)
        for s, ids in self.idx.items():
            for (iq, _), qi in zip(ids, q_urdf.get(s, np.zeros(7))):
                q[iq] = qi
        g = self.pin.computeGeneralizedGravity(self.model, self.data, q)
        return {s: np.array([g[iv] for _, iv in ids]) for s, ids in self.idx.items()}
