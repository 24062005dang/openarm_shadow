"""Backend MuJoCo: robot OpenArm v1.0 mô phỏng vật lý (MJCF của enactic/openarm_mujoco, thư mục openarm_mujoco/v1).

Thay cho SimRobot (đo = lệnh) khi muốn xem robot 3D thật, có động lực học, giới hạn khớp và va chạm của mô hình.
Góc khớp MJCF v1 trùng góc URDF dùng trong kinematics.py (cùng trục, dấu, zero; FK lệch 0 mm), nên lệnh từ
SafetyGate đưa thẳng vào, không qua urdf_to_motor.

- mode physics: lệnh vào actuator position của MJCF (kp/kv giống gain v1.0), mỗi send() bước vật lý đúng
  1/control_hz giây. read() trả góc thật trong mô phỏng (trễ / võng so với lệnh như robot thật).
  gravity_comp: cộng qfrc_bias vào các khớp tay (không bù thì J1 võng ~5° khi tay đưa ra trước).
- mode kinematic: ghi thẳng qpos = lệnh (như SimRobot, chỉ thêm hình 3D).
Kẹp: lệnh 0..1 (0 = đóng) -> hành trình ngón 0..gripper_stroke_m.

Cửa sổ MuJoCo chạy ở TIẾN TRÌNH RIÊNG (chỉ đọc qpos qua bộ nhớ chung, vẽ 60 Hz): viewer passive (GLFW, luồng nền)
chạy chung tiến trình với cửa sổ OpenCV (Qt) bị treo, và core dump lúc thoát. Kéo robot bằng chuột trong cửa sổ
MuJoCo vì thế không tác động vào mô phỏng.
"""
from __future__ import annotations

import multiprocessing as mp
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _viewer_main(model_path, qpos, stop, fps=30.0, shadow_size=2048):
    """Tiến trình viewer: chép qpos chung vào bản sao mô hình, tính FK, vẽ. Dừng khi stop, đóng cửa sổ hoặc
    tiến trình chính chết (bị kill: không để cửa sổ mồ côi).

    Viewer dùng chung GPU với MediaPipe (delegate gpu): scene.xml đặt bóng 8192 px, vẽ 60 fps làm Hand chậm
    33 -> 58 ms/lệnh, fusion 3 camera tụt 20 -> 10 fps. Mặc định 30 fps, bóng 2048 px (robot.mujoco.viewer_fps,
    viewer_shadow_size)."""
    import os
    import signal

    import mujoco
    import mujoco.viewer
    signal.signal(signal.SIGINT, signal.SIG_IGN)     # Ctrl+C do tiến trình chính xử lý (về nghỉ, đóng viewer)
    parent = mp.parent_process()
    m = mujoco.MjModel.from_xml_path(model_path)
    m.vis.quality.shadowsize = int(shadow_size)
    d = mujoco.MjData(m)
    buf = np.frombuffer(qpos.get_obj())
    v = mujoco.viewer.launch_passive(m, d)
    while v.is_running() and not stop.is_set() and (parent is None or parent.is_alive()):
        with qpos.get_lock():
            d.qpos[:] = buf
        mujoco.mj_kinematics(m, d)
        v.sync()
        time.sleep(1.0 / fps)
    # Thoát thẳng, không qua v.close() / glfw.terminate(): bước dọn GLFW đó segfault khi app dừng bằng Ctrl+C.
    # Cửa sổ và tài nguyên GPU được hệ điều hành thu hồi khi tiến trình kết thúc.
    os._exit(0)


class MujocoRobot:
    def __init__(self, cfg: dict, sides):
        import mujoco
        self.mj = mujoco
        rcfg = cfg["robot"]
        mc = dict(rcfg.get("mujoco") or {})
        path = Path(mc.get("model", "openarm_mujoco/v1/scene.xml"))
        path = path if path.is_absolute() else ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Không thấy mô hình MuJoCo {path} (robot.mujoco.model)")
        self.sides = list(sides)
        self.mode = mc.get("mode", "physics")
        if self.mode not in ("physics", "kinematic"):
            raise ValueError(f"robot.mujoco.mode phải là physics hoặc kinematic, không phải {self.mode}")
        self.gravity_comp = bool(mc.get("gravity_comp", True))
        self.show_viewer = bool(mc.get("viewer", True))
        self.viewer_opts = (float(mc.get("viewer_fps", 30)), int(mc.get("viewer_shadow_size", 2048)))
        self.stroke = float(mc.get("gripper_stroke_m", 0.044))
        self.dt = 1.0 / float(rcfg["control_hz"])
        self.rest = np.deg2rad(np.asarray(rcfg.get("rest_pose_deg", [0] * 7), float))
        self.path = str(path)
        self.m = mujoco.MjModel.from_xml_path(self.path)
        self.d = mujoco.MjData(self.m)
        self.n_sub = max(1, int(round(self.dt / self.m.opt.timestep)))
        # Chỉ số theo TÊN (thứ tự actuator trong MJCF là trái rồi phải). Có đủ 2 tay dù chỉ điều khiển 1 tay:
        # tay không điều khiển giữ tư thế nghỉ.
        self.qadr, self.dadr, self.act, self.fadr = {}, {}, {}, {}
        for s in ("left", "right"):
            jids = [self.m.joint(f"openarm_{s}_joint{i}").id for i in range(1, 8)]
            self.qadr[s] = np.array([self.m.jnt_qposadr[j] for j in jids])
            self.dadr[s] = np.array([self.m.jnt_dofadr[j] for j in jids])
            self.act[s] = np.array([self.m.actuator(f"{s}_joint{i}_ctrl").id for i in range(1, 8)]
                                   + [self.m.actuator(f"{s}_finger1_ctrl").id])
            self.fadr[s] = np.array([self.m.jnt_qposadr[self.m.joint(f"openarm_{s}_finger_joint{k}").id]
                                     for k in (1, 2)])
        self.arm_dofs = np.concatenate([self.dadr[s] for s in ("left", "right")])
        self.lower = {s: self.m.jnt_range[[self.m.joint(f"openarm_{s}_joint{i}").id for i in range(1, 8)], 0]
                      for s in ("left", "right")}
        self.upper = {s: self.m.jnt_range[[self.m.joint(f"openarm_{s}_joint{i}").id for i in range(1, 8)], 1]
                      for s in ("left", "right")}
        self._apply_pd(mc)
        self._warn_soft_limits(cfg.get("safety", {}).get("soft_limits_deg", {}))
        self.lock = threading.Lock()          # send() chạy ở luồng điều khiển, read() ở luồng chính
        self.viewer = None                    # tiến trình viewer
        self._qpos_shared = self._stop = None
        self.enabled = False

    def _apply_pd(self, mc):
        """robot.mujoco.kp / kv (7 giá trị J1-J7, dùng cho cả hai tay): ghi đè gain PD của actuator position trong MJCF
        (lực = kp (ctrl - q) - kv dq). Giới hạn mô-men (forcerange) của MJCF giữ nguyên nên tăng gain không làm vượt
        sức motor. Không đặt thì dùng gain của MJCF (= gain chính thức v1.0)."""
        kp, kv = mc.get("kp"), mc.get("kv")
        if kp is None and kv is None:
            return
        kp = np.asarray(kp if kp is not None else [self.m.actuator_gainprm[a, 0] for a in self.act["right"][:7]], float)
        kv = np.asarray(kv if kv is not None else [-self.m.actuator_biasprm[a, 2] for a in self.act["right"][:7]], float)
        if kp.shape != (7,) or kv.shape != (7,) or np.any(kp <= 0) or np.any(kv < 0):
            raise ValueError("robot.mujoco.kp / kv phải là 7 số (kp > 0, kv >= 0)")
        for s in ("left", "right"):
            for k, a in enumerate(self.act[s][:7]):
                self.m.actuator_gainprm[a, 0] = kp[k]
                self.m.actuator_biasprm[a, 1] = -kp[k]
                self.m.actuator_biasprm[a, 2] = -kv[k]

    def _warn_soft_limits(self, soft):
        """Giới hạn mềm SafetyGate rộng hơn giới hạn khớp MJCF -> MuJoCo kẹp lệnh, robot lệch khỏi lệnh."""
        for s in self.sides:
            if s not in soft:
                continue
            lim = np.deg2rad(np.asarray(soft[s], float))
            bad = [i + 1 for i in range(7) if lim[i, 0] < self.lower[s][i] - 1e-3 or lim[i, 1] > self.upper[s][i] + 1e-3]
            if bad:
                print(f"Cảnh báo MuJoCo: safety.soft_limits_deg {s} J{bad} rộng hơn giới hạn MJCF "
                      f"{np.round(np.rad2deg(self.lower[s]), 1)}..{np.round(np.rad2deg(self.upper[s]), 1)} độ "
                      "(dùng config/mujoco_sim.yaml)")

    # ------------------------------------------------------------------
    def _set_pose(self, s, q8):
        """Ghi thẳng tư thế (7 góc + kẹp 0..1) vào qpos và ctrl của một tay."""
        q8 = np.asarray(q8, float)
        self.d.qpos[self.qadr[s]] = q8[:7]
        self.d.qpos[self.fadr[s]] = self.stroke * np.clip(q8[7], 0.0, 1.0)
        self.d.ctrl[self.act[s]] = np.append(q8[:7], self.stroke * np.clip(q8[7], 0.0, 1.0))

    def _publish(self):
        """Gửi qpos hiện tại cho tiến trình viewer."""
        if self._qpos_shared is not None:
            with self._qpos_shared.get_lock():
                np.frombuffer(self._qpos_shared.get_obj())[:] = self.d.qpos

    def connect(self):
        """Đặt cả hai tay ở tư thế nghỉ (kẹp mở một nửa như SimRobot), mở cửa sổ MuJoCo."""
        self.mj.mj_resetData(self.m, self.d)
        for s in ("left", "right"):
            self._set_pose(s, np.append(self.rest, 0.5))
        self.mj.mj_forward(self.m, self.d)
        if self.show_viewer and self.viewer is None:
            ctx = mp.get_context("spawn")     # không fork tiến trình đang có MediaPipe / OpenCV / luồng camera
            self._qpos_shared = ctx.Array("d", self.m.nq)
            self._stop = ctx.Event()
            self._publish()
            self.viewer = ctx.Process(target=_viewer_main, args=(self.path, self._qpos_shared, self._stop, *self.viewer_opts),
                                      name="mujoco-viewer", daemon=True)
            self.viewer.start()
        return self.read()

    def read(self):
        with self.lock:
            return {s: np.append(self.d.qpos[self.qadr[s]],
                                 np.clip(self.d.qpos[self.fadr[s][0]] / self.stroke, 0.0, 1.0)) for s in self.sides}

    def enable(self):
        self.enabled = True

    def send(self, cmd, dq=None):
        """cmd[side]: 7 góc URDF + kẹp 0..1 (NaN = giữ). dq (feedforward vận tốc của robot thật) bị bỏ qua."""
        with self.lock:
            for s in self.sides:
                c = np.asarray(cmd[s], float)
                if self.mode == "kinematic":
                    cur = np.append(self.d.qpos[self.qadr[s]], self.d.qpos[self.fadr[s][0]] / self.stroke)
                    self._set_pose(s, np.where(np.isfinite(c), c, cur))
                    continue
                ctrl = self.d.ctrl[self.act[s]]
                ctrl[:7] = np.where(np.isfinite(c[:7]), c[:7], ctrl[:7])
                if np.isfinite(c[7]):
                    ctrl[7] = self.stroke * np.clip(c[7], 0.0, 1.0)
                self.d.ctrl[self.act[s]] = ctrl
            if self.mode == "kinematic":
                self.d.qvel[:] = 0.0
                self.mj.mj_forward(self.m, self.d)
            else:
                for _ in range(self.n_sub):
                    if self.gravity_comp:
                        self.d.qfrc_applied[self.arm_dofs] = self.d.qfrc_bias[self.arm_dofs]
                    self.mj.mj_step(self.m, self.d)
            self._publish()

    def relax(self):
        self.enabled = False

    def close(self):
        self.enabled = False
        if self.viewer is not None:
            self._stop.set()
            self.viewer.join(timeout=3.0)
            if self.viewer.is_alive():
                self.viewer.terminate()
            self.viewer = None
