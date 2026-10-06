"""Backend MuJoCo (openarm_mujoco/v1): góc MJCF = góc URDF, bám lệnh, kẹp, NaN = giữ, cảnh báo giới hạn."""
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from openarm_shadow.config import ROOT, load_config
from openarm_shadow.core.kinematics import ArmKinematics
from openarm_shadow.robot import make_robot
from openarm_shadow.safety.gate import SafetyGate

MODEL = ROOT / "openarm_mujoco" / "v1" / "scene.xml"
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="chưa có openarm_mujoco/v1")


def make(mode="physics", sides=("right", "left"), extra=None):
    cfg = load_config([ROOT / "config" / "mujoco_sim.yaml"] + list(extra or []))
    cfg["robot"]["mujoco"].update(mode=mode, viewer=False)
    robot = make_robot("mujoco", cfg, list(sides))
    return cfg, robot


def test_mjcf_fk_matches_urdf_kinematics():
    _, robot = make()
    rng = np.random.default_rng(0)
    for s in ("right", "left"):
        kin = ArmKinematics(s)
        for _ in range(50):
            q = rng.uniform(robot.lower[s], robot.upper[s])
            robot.d.qpos[robot.qadr[s]] = q
            mujoco.mj_kinematics(robot.m, robot.d)
            fr = kin.frames(q)
            for i in range(1, 8):
                b = robot.m.body(f"openarm_{s}_link{i}").id
                assert np.allclose(robot.d.xpos[b], fr[i][1], atol=1e-6)
                assert np.allclose(robot.d.xmat[b].reshape(3, 3), fr[i][0], atol=1e-6)


def test_physics_tracks_command_with_gravity_comp():
    cfg, robot = make()
    q0 = robot.connect()
    assert np.allclose(q0["right"][:7], 0.0) and np.isclose(q0["right"][7], 0.5)
    cmd = {"right": np.append(np.deg2rad([30, 20, 0, 90, 0, 0, 0]), 1.0),
           "left": np.append(np.deg2rad([-30, -20, 0, 90, 0, 0, 0]), 0.0)}
    for _ in range(int(2.0 * cfg["robot"]["control_hz"])):
        robot.send(cmd)
    q = robot.read()
    for s in ("right", "left"):
        assert np.max(np.rad2deg(np.abs(q[s][:7] - cmd[s][:7]))) < 0.5
    assert q["right"][7] > 0.95 and q["left"][7] < 0.05
    robot.close()


def test_kinematic_mode_and_nan_holds_joint():
    _, robot = make("kinematic", sides=("right",))
    robot.connect()
    target = np.append(np.deg2rad([10, 30, -20, 60, 15, -10, 25]), 0.3)
    robot.send({"right": target})
    assert np.allclose(robot.read()["right"], target)
    held = target.copy()
    held[3] = np.nan
    held[0] = np.deg2rad(40)
    robot.send({"right": held})
    q = robot.read()["right"]
    assert np.isclose(q[3], target[3]) and np.isclose(q[0], np.deg2rad(40))
    assert set(robot.read()) == {"right"}


def test_soft_limit_warning(capsys):
    cfg = load_config()
    cfg["robot"]["mujoco"]["viewer"] = False
    make_robot("mujoco", cfg, ["right"])
    assert "J[4" in capsys.readouterr().out          # default.yaml: J4 tới 135° > 131.8° của MJCF
    make()
    assert "Cảnh báo MuJoCo" not in capsys.readouterr().out


def test_safety_gate_drives_mujoco_robot():
    cfg, robot = make()
    sides = ["right", "left"]
    kins = {s: ArmKinematics(s) for s in sides}
    gate = SafetyGate(kins, cfg["safety"])
    gate.reset(robot.connect())
    gate.engage(0.0)
    goal = {"right": np.append(np.deg2rad([40, 20, 10, 60, 10, 10, 5]), 0.8),
            "left": np.append(np.deg2rad([-40, -20, -10, 60, -10, 10, -5]), 0.8)}
    dt = 1.0 / cfg["robot"]["control_hz"]
    for k in range(int(5.0 / dt)):
        t = k * dt
        gate.set_target(goal, t)
        robot.send(gate.step(dt, t))
    q = robot.read()
    for s in sides:
        assert gate.status == "follow"
        assert np.max(np.rad2deg(np.abs(q[s][:7] - goal[s][:7]))) < 1.0


def test_pd_override_and_defaults():
    _, default = make(extra=[])
    a = default.act["right"][4]
    cfg = load_config([ROOT / "config" / "mujoco_sim.yaml"])
    assert cfg["robot"]["mujoco"]["kp"][4] == 25 and cfg["safety"]["velocity_tracking"]["enabled"]
    assert np.isclose(default.m.actuator_gainprm[a, 0], 25) and np.isclose(default.m.actuator_biasprm[a, 1], -25)
    assert np.isclose(default.m.actuator_biasprm[a, 2], -0.8)
    assert np.isclose(default.m.actuator_gainprm[default.act["left"][0], 0], 70)          # áp cho cả hai tay
    cfg["robot"]["mujoco"].pop("kp")
    cfg["robot"]["mujoco"].pop("kv")
    cfg["robot"]["mujoco"]["viewer"] = False
    stock = make_robot("mujoco", cfg, ["right"])
    assert np.isclose(stock.m.actuator_gainprm[stock.act["right"][4], 0], 10)              # bỏ kp/kv: gain MJCF
    cfg["robot"]["mujoco"]["kp"] = [70, 70, 70, 60, 25, 25]
    import pytest
    with pytest.raises(ValueError):
        make_robot("mujoco", cfg, ["right"])
