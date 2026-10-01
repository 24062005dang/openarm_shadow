import numpy as np


class SimRobot:
    """Robot 'lý tưởng': đo được = lệnh vừa gửi. Dùng để thử pipeline + vẽ, không cần phần cứng."""

    def __init__(self, sides, q0=None):
        self.sides = list(sides)
        self.q = {s: np.zeros(8) if q0 is None else np.asarray(q0[s], float).copy() for s in self.sides}
        for s in self.sides:
            self.q[s][7] = 0.5

    def connect(self):
        return self.read()

    def read(self):
        return {s: v.copy() for s, v in self.q.items()}

    def enable(self):
        pass

    def send(self, cmd, dq=None):
        for s in self.sides:
            self.q[s] = np.asarray(cmd[s], float).copy()

    def relax(self):
        pass

    def close(self):
        pass
