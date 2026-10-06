"""models.delegate: GPU không khởi tạo được -> tự quay về CPU (không chết)."""
import types

import pytest


def test_gpu_falls_back_to_cpu(tmp_path, monkeypatch, capsys):
    pytest.importorskip("mediapipe")
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision
    from openarm_shadow.vision.perception import Perception

    made = []

    def fake_create(name):
        def create(opts):
            if opts.base_options.delegate == mpt.BaseOptions.Delegate.GPU:
                raise RuntimeError("no EGL")
            made.append(name)
            return types.SimpleNamespace(close=lambda: None)
        return create

    monkeypatch.setattr(vision.PoseLandmarker, "create_from_options", staticmethod(fake_create("pose")))
    monkeypatch.setattr(vision.HandLandmarker, "create_from_options", staticmethod(fake_create("hand")))
    pm, hm = tmp_path / "pose.task", tmp_path / "hand.task"
    pm.write_bytes(b"x")
    hm.write_bytes(b"x")
    p = Perception(str(pm), str(hm), delegate="gpu", parallel=False)
    assert p.delegate == {"pose": "cpu", "hand": "cpu"} and made == ["pose", "hand"]
    assert "không chạy được trên GPU" in capsys.readouterr().out
    p2 = Perception(str(pm), str(hm), delegate="auto", parallel=False)
    assert p2.delegate == {"pose": "cpu", "hand": "cpu"}
