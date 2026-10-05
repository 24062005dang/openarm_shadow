"""Luồng perception: đọc camera + nhận diện ở nền, luôn giữ kết quả mới nhất."""
from __future__ import annotations

import threading


class PerceptionWorker(threading.Thread):
    """Đọc camera + MediaPipe ở luồng riêng: luồng chính vẽ/điều khiển khung N trong lúc khung N+1 đang được nhận
    diện (trước đây làm nối tiếp nên vẽ chặn nhận diện). Luôn giữ kết quả MỚI NHẤT; luồng chính chậm thì bỏ khung cũ."""

    def __init__(self, cap, process):
        super().__init__(daemon=True, name="perception")
        self.cap, self.process = cap, process
        self.cond = threading.Condition()
        self.latest, self.seq, self.taken = None, 0, 0
        self.running, self.done, self.error = True, False, None

    def run(self):
        try:
            while self.running:
                ok, sample = self.cap.read()
                if not ok:
                    self.error = getattr(self.cap, "error", None) or "Nguồn video hết khung hoặc mất kết nối."
                    break
                item = (sample, *self.process(sample))
                with self.cond:
                    self.latest, self.seq = item, self.seq + 1
                    self.cond.notify_all()
        except BaseException as e:           # lỗi nhận diện: báo cho luồng chính, không chết im lặng
            self.error = f"{type(e).__name__}: {e}"
        finally:
            with self.cond:
                self.done = True
                self.cond.notify_all()

    def get(self, timeout=10.0):
        """Kết quả mới (sample, frame, extra) chưa lấy; None nếu luồng đã dừng (xem .error) hoặc quá timeout."""
        with self.cond:
            if not self.cond.wait_for(lambda: self.seq > self.taken or self.done, timeout):
                self.error = f"không có khung mới trong {timeout:.0f} s"
                return None
            if self.seq <= self.taken:
                return None
            self.taken = self.seq
            return self.latest

    def stop(self):
        self.running = False
        self.join(timeout=5.0)               # cap.read() có thể chờ tới 3 s
