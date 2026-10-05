"""Phát lại dữ liệu thô (cameras.raw_replay) và tự engage (runtime.session), không cần camera."""
import csv

import cv2
import numpy as np
import pytest
import yaml

from openarm_shadow.cameras import RawReplaySource
from openarm_shadow.runtime.session import AutoEngage


def write_camera(root, name, times, depth=False, size=(64, 48)):
    """Ghi một camera giả theo đúng định dạng record_multicam_raw.py; khung k có mức xám = k."""
    d = root / name
    d.mkdir(parents=True)
    w, h = size
    vw = cv2.VideoWriter(str(d / "rgb.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (w, h))
    for k in range(len(times)):
        vw.write(np.full((h, w, 3), 10 * k, np.uint8))
    vw.release()
    with (d / "timestamps.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "host_arrival_s", "device_timestamp_s"])
        for k, t in enumerate(times):
            wr.writerow([k, f"{t:.6f}", ""])
    meta = {"rgb_width": w, "rgb_height": h, "depth_width": None, "depth_height": None}
    if depth:
        np.stack([np.full((h, w), 1000 + k, "<u2") for k in range(len(times))]).tofile(d / "depth_u16_mm.raw")
        meta.update(depth_width=w, depth_height=h)
    return meta


@pytest.fixture
def recording(tmp_path):
    ref_t = 100.0 + np.arange(6) / 30
    side_t = np.delete(ref_t + 0.010, [3, 4])   # camera phụ lệch 10 ms và mất 2 khung liền (3, 4)
    meta = {"cameras": {"front": write_camera(tmp_path, "front", ref_t),
                        "side": write_camera(tmp_path, "side", side_t, depth=True)}}
    (tmp_path / "metadata.yaml").write_text(yaml.safe_dump(meta))
    (tmp_path / "intrinsics.yaml").write_text(yaml.safe_dump(
        {"cameras": {"side": {"fx": 50.0, "fy": 50.0, "ppx": 32.0, "ppy": 24.0}}}))
    return tmp_path


def read_all(src):
    out = []
    while True:
        ok, s = src.read()
        if not ok:
            return out
        out.append(s)


def test_pairs_nearest_frame_and_marks_stale(recording):
    src = RawReplaySource(recording, ["front", "side"], max_skew_s=0.04)
    samples = read_all(src)
    src.close()
    assert len(samples) == 6
    for k, s in enumerate(samples):
        assert s.t == pytest.approx(100.0 + k / 30)
        assert abs(int(s.views[0].bgr.mean()) - 10 * k) <= 2
        assert s.views[1].intrinsics["fx"] == 50.0
    assert [s.stale[1] for s in samples] == [False, False, False, False, True, False]
    # Khung tham chiếu 3: khung phụ gần nhất lệch ~23 ms -> vẫn ghép; khung 4: gần nhất lệch ~43 ms > 40 ms -> stale
    assert samples[3].skew_s == pytest.approx(1 / 30 - 0.010, abs=1e-6)
    assert samples[0].views[1].depth_m[0, 0] == pytest.approx(1.000)
    assert samples[1].skew_s == pytest.approx(0.010, abs=1e-6)


def test_start_stride_and_frame_limit(recording):
    src = RawReplaySource(recording, ["front", "side"], start_s=1 / 30 - 1e-6, stride=2, max_frames=2)
    samples = read_all(src)
    src.close()
    assert [round((s.t - 100.0) * 30) for s in samples] == [2, 4]


def test_depth_camera_without_intrinsics_is_rejected(recording):
    (recording / "intrinsics.yaml").unlink()
    with pytest.raises(SystemExit):
        RawReplaySource(recording, ["front", "side"])


def test_auto_engage_needs_continuous_ready():
    a = AutoEngage(2.0)
    assert not a.update(True, False, 0.0) and a.countdown == pytest.approx(2.0)
    assert not a.update(False, False, 1.5) and a.countdown is None      # mất READY: đếm lại
    assert not a.update(True, False, 2.0)
    assert a.update(True, False, 4.0)                                   # đủ 2 s liên tục
    assert not a.update(True, False, 9.0)                               # chỉ một lần


def test_auto_engage_off_and_manual_override():
    assert not AutoEngage(None).update(True, False, 100.0)
    a = AutoEngage(1.0)
    a.update(True, False, 0.0)
    a.manual()                                                          # bấm SPACE: thôi tự engage
    assert a.countdown is None and not a.update(True, False, 5.0)
