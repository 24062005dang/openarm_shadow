"""Luồng nhận diện nền của app (PerceptionWorker): khung mới nhất, báo mất camera, không nuốt lỗi."""
import threading
import time

from openarm_shadow.app import PerceptionWorker


class FakeCap:
    def __init__(self, n, delay=0.0):
        self.i, self.n, self.delay, self.error = 0, n, delay, None

    def read(self):
        time.sleep(self.delay)
        if self.i >= self.n:
            self.error = "het khung"
            return False, None
        self.i += 1
        return True, self.i


def test_worker_gives_each_result_once_and_reports_lost_camera():
    w = PerceptionWorker(FakeCap(3, delay=0.02), lambda s: (s * 10, None))
    w.start()
    got = []
    while (item := w.get(timeout=2.0)) is not None:
        got.append(item[:2])
    w.stop()
    assert got and got == sorted(set(got))           # không lấy trùng một kết quả
    assert got[-1] == (3, 30)                        # khung cuối cùng không bị mất
    assert w.error == "het khung"


def test_slow_consumer_skips_to_latest_frame():
    """Luồng chính chậm hơn camera: bỏ khung cũ, không xếp hàng (độ trễ không tăng dần)."""
    w = PerceptionWorker(FakeCap(50, delay=0.005), lambda s: (s, None))
    w.start()
    first = w.get(timeout=2.0)
    time.sleep(0.15)
    second = w.get(timeout=2.0)
    w.stop()
    assert second[0] - first[0] > 5


def test_processing_error_stops_loop_with_message():
    def boom(s):
        raise ValueError("model hong")
    w = PerceptionWorker(FakeCap(5), boom)
    w.start()
    assert w.get(timeout=2.0) is None
    w.stop()
    assert "model hong" in w.error


def test_stop_ends_thread():
    w = PerceptionWorker(FakeCap(10_000, delay=0.001), lambda s: (s, None))
    w.start()
    w.get(timeout=2.0)
    w.stop()
    assert not w.is_alive() and threading.active_count() >= 1
