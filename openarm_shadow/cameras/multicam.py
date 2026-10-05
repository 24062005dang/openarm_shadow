"""Nhiều camera chạy song song, đồng bộ bằng phần mềm (mỗi camera một luồng đọc, ghép khung theo thời điểm).

Ý tưởng từ stereohand (MIT): mỗi camera một luồng chụp, ghép khung theo thời điểm."""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

REALSENSE_NAMES = {"realsense", "rs", "d455", "d435", "d435i"}


@dataclass
class MultiSample:
    views: list                 # CameraSample theo thứ tự fusion.cameras
    t: float                    # thời điểm (s, time.monotonic) của khung camera tham chiếu
    skew_s: float = 0.0         # lệch thời gian lớn nhất giữa các khung được ghép (không tính camera stale)
    stale: list = None          # stale[i] = True: camera i không có khung đủ mới (treo, mất kết nối) -> bỏ qua
    lags: list = None           # lags[i]: lệch thời gian (s) khung camera i so với khung tham chiếu


class MultiCameraSource:
    """Mỗi camera một luồng đọc; read() lấy khung của camera tham chiếu (mới nhất, hoặc theo camera chậm nhất với
    fusion.sync = slowest) và khung gần thời điểm nhất của mỗi camera còn lại. D455/D435i và webcam không đồng bộ
    phần cứng được với nhau."""

    sync, sync_max_wait = "latest", 0.15       # mặc định (ghi đè trong __init__ theo fusion.sync)

    def __init__(self, cfg):
        from .sources import OpenCVSource, RealSenseSource, webcam_options
        fc = cfg["fusion"]
        self.names = [c["name"] for c in fc["cameras"]]
        self.kinds = []
        self.tol = float(fc.get("sync_tol_s", 0.025))
        # Khung của camera phụ cũ hơn mức này so với camera tham chiếu = camera đó đang treo: không dùng khung cũ.
        # Khung camera phụ lệch thời gian quá max_skew_s so với khung tham chiếu thì KHÔNG ghép (coi như camera đó
        # không có khung lần này). Trước đây chỉ loại khi > 100 ms, nên khung lệch 40-60 ms vẫn bị triangulate
        # chung: tay đang di chuyển 1 m/s lệch 4-6 cm giữa hai ảnh.
        self.stale_s = float(fc.get("max_skew_s", fc.get("stale_s", max(1.6 * self.tol, 0.04))))
        # Chờ tối đa bao lâu cho khung khớp của camera phụ; 0 = không chờ, lấy khung gần nhất đang có (không khựng)
        self.pair_wait = float(fc.get("pair_wait_s", self.tol))
        # sync "latest": khung tham chiếu MỚI NHẤT, camera phụ lấy khung gần nhất đang có (không chờ) -> ít trễ, nhưng
        # camera đến chậm (iPhone/DroidCam ~54 ms) gần như luôn lệch > max_skew_s -> bị bỏ.
        # sync "slowest": chọn khung tham chiếu mới nhất mà mọi camera đang chạy đều ĐÃ có khung cùng thời điểm -> mọi
        # camera cùng góp, đổi lại trễ thêm bằng camera chậm nhất. Camera chậm hơn sync_max_wait_s (treo) thì không chờ.
        self.sync = str(fc.get("sync", "latest"))
        self.sync_max_wait = float(fc.get("sync_max_wait_s", 0.15))
        n_rs = sum(str(c.get("source", "realsense")).lower() in REALSENSE_NAMES for c in fc["cameras"])
        self.srcs = []
        try:
            for c in fc["cameras"]:
                src = str(c.get("source", "realsense")).lower()
                if src in REALSENSE_NAMES:
                    if str(c.get("serial") or "").startswith("SERIAL_"):
                        raise SystemExit(f"Camera '{c['name']}': chưa điền serial thật (đang là {c['serial']}).\n"
                                         "Xem serial: python scripts/list_cameras.py")
                    if n_rs > 1 and not c.get("serial"):
                        raise SystemExit("Có nhiều RealSense: đặt 'serial' cho từng camera trong fusion.cameras.\n"
                                         "Xem serial: python scripts/list_cameras.py")
                    rcfg = dict(cfg["camera"].get("realsense", {}), serial=c.get("serial"))
                    self.srcs.append(RealSenseSource({"camera": {"realsense": rcfg}}))
                    self.kinds.append("realsense")
                else:
                    self.srcs.append(OpenCVSource(c["source"], **webcam_options(cfg)))
                    self.kinds.append("opencv")
                    print(f"Camera '{c['name']}' (webcam {c['source']}): {self.srcs[-1].mode}")
        except BaseException:
            self.close()
            raise
        # Độ trễ cố định của từng camera (s): webcam laptop thường trả khung chậm hơn RealSense vài chục ms.
        self.latency = [float(c.get("latency_s", 0.0)) for c in fc["cameras"]]
        self.buf = [deque(maxlen=12) for _ in self.srcs]     # 12 khung ~0,4 s: đủ chờ camera chậm (sync slowest)
        self.stats = [{"frames": 0, "fails": 0, "error": None} for _ in self.srcs]   # để báo lỗi khi mất camera
        self.error = None
        self.cond = threading.Condition()
        self.running = True
        self.last_ref_t = -1.0
        self.threads = [threading.Thread(target=self._loop, args=(i,), daemon=True) for i in range(len(self.srcs))]
        for th in self.threads:
            th.start()

    def _loop(self, i):
        st = self.stats[i]
        while self.running:
            try:
                ok, s = self.srcs[i].read()
            except Exception as e:           # vd RealSense "Frame didn't arrive": thử tiếp, ghi lại để báo
                ok, st["error"] = False, f"{type(e).__name__}: {e}"
            t = time.monotonic() - self.latency[i]
            if not ok:
                st["fails"] += 1
                time.sleep(0.005)
                continue
            st["frames"] += 1
            s.bgr = np.ascontiguousarray(s.bgr).copy()   # không giữ buffer của driver
            with self.cond:
                self.buf[i].append((t, s))
                self.cond.notify_all()

    def _pick_synced_ref(self):
        """sync slowest: khung tham chiếu mới nhất (chưa dùng) có thời điểm <= khung mới nhất của camera phụ chậm nhất
        (+ sync_tol_s). Camera phụ chưa có khung nào thì chờ; tụt sau khung tham chiếu mới nhất hơn sync_max_wait_s
        (treo/mất) thì không chờ (sẽ bị đánh dấu stale). -> (t, sample) hoặc None (chờ thêm)."""
        fresh = [e for e in self.buf[0] if e[0] > self.last_ref_t]
        if not fresh:
            return None
        newest = fresh[-1][0]
        t_ok = np.inf
        for b in self.buf[1:]:
            if not b:
                return None
            if newest - b[-1][0] <= self.sync_max_wait:
                t_ok = min(t_ok, b[-1][0] + self.tol)
        cand = [e for e in fresh if e[0] <= t_ok]
        if cand:
            return cand[-1]
        # Khung tham chiếu cũ nhất chưa dùng đã chờ quá sync_max_wait mà camera phụ vẫn chưa tới: thôi chờ
        return fresh[-1] if newest - fresh[0][0] > self.sync_max_wait else None

    def read(self, timeout=3.0):
        deadline = time.monotonic() + timeout
        with self.cond:
            while True:
                if self.sync == "slowest" and len(self.buf) > 1:
                    picked = self._pick_synced_ref()
                elif self.buf[0] and self.buf[0][-1][0] > self.last_ref_t:
                    picked = self.buf[0][-1]
                else:
                    picked = None
                if picked is not None:
                    break
                left = deadline - time.monotonic()
                if left <= 0 or not self.running:
                    self.error = self.describe_stall(timeout)
                    return False, None
                self.cond.wait(left)
            t_ref, ref = picked
            self.last_ref_t = t_ref
            views, skew, stale, lags = [ref], 0.0, [False], [0.0]
            for i in range(1, len(self.buf)):
                wait_until = time.monotonic() + self.pair_wait
                while True:
                    cands = list(self.buf[i])
                    best = min(cands, key=lambda ts: abs(ts[0] - t_ref)) if cands else None
                    # Đủ gần, hoặc đã có khung mới hơn t_ref (chờ thêm chỉ có khung mới hơn nữa) -> thôi chờ
                    newer = best is not None and (abs(best[0] - t_ref) <= self.tol or
                                                  any(ts[0] >= t_ref for ts in cands))
                    # Chưa có khung nào (vd RealSense đang khởi động): chờ tới hết timeout, không theo pair_wait
                    left = (wait_until if cands else deadline) - time.monotonic()
                    if newer or left <= 0 or not self.running:
                        break
                    self.cond.wait(left)
                if best is None:
                    self.error = self.describe_stall(timeout, self.names[i])
                    return False, None
                views.append(best[1])
                lag = abs(best[0] - t_ref)
                lags.append(lag)
                stale.append(lag > self.stale_s)
                if lag <= self.stale_s:
                    skew = max(skew, lag)
        return True, MultiSample(views, t_ref, skew, stale, lags)

    def describe_stall(self, timeout, name=None):
        """Câu báo lỗi khi một camera (mặc định camera tham chiếu) không gửi khung mới: số khung/lỗi từng camera."""
        parts = []
        for cam_name, st in zip(self.names, self.stats):
            parts.append(f"  {cam_name}: {st['frames']} khung, {st['fails']} lần đọc lỗi"
                         + (f", lỗi cuối: {st['error']}" if st["error"] else ""))
        return (f"Camera '{name or self.names[0]}' không gửi khung mới trong {timeout:.0f} s.\n" + "\n".join(parts) +
                "\nKiểm tra: camera có bị app khác chiếm không, cáp/cổng USB, chỉ số webcam (list_cameras.py).")

    def close(self):
        self.running = False
        for th in getattr(self, "threads", []):     # chờ luồng đọc thoát trước khi giải phóng camera
            th.join(timeout=1.0)
        for s in self.srcs:
            try:
                s.close()
            except Exception:
                pass
