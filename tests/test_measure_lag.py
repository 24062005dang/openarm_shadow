"""scripts/measure_lag.py tìm đúng độ trễ đã biết."""
import runpy
import sys

import numpy as np


def test_measure_lag_finds_known_delays(tmp_path, capsys):
    t = np.arange(0, 10, 0.01)
    sig = lambda tt: np.deg2rad(30) * np.sin(2 * np.pi * 0.4 * tt)
    tgt_t = np.arange(0, 10, 1 / 15)
    tgt = np.zeros((len(tgt_t), 8)); tgt[:, 4] = sig(tgt_t)
    cmd = np.zeros((len(t), 8)); cmd[:, 4] = sig(t - 0.05)           # SafetyGate trễ 50 ms
    meas = np.zeros((len(t), 8)); meas[:, 4] = sig(t - 0.12)         # motor trễ thêm 70 ms
    f = tmp_path / "run.npz"
    np.savez(f, t=tgt_t, target_right=tgt, ctl_t=t, ctl_cmd_right=cmd, ctl_meas_right=meas)
    sys.argv = ["measure_lag.py", str(f)]
    runpy.run_path("scripts/measure_lag.py", run_name="__main__")
    out = capsys.readouterr().out
    row = [ln for ln in out.splitlines() if ln.strip().startswith("J5")][0]
    nums = [float(x) for x in row.replace("°", " ").replace("ms", " ").replace("(", " ").replace(")", " ").split()[1:]]
    # [biên độ, mục tiêu->lệnh, lệnh->đo, sai số, mục tiêu->đo, sai số]
    assert abs(nums[1] - 50) <= 10 and abs(nums[2] - 70) <= 10 and abs(nums[4] - 120) <= 10, row
