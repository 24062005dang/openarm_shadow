#!/usr/bin/env python3
"""Bước 2: chuyển động đầu tiên. Giữ J1-J6 tại chỗ, lắc J7 (xoay cổ tay) ±amp độ.

ĐIỀU KIỆN TRƯỚC KHI CHẠY
- Tay đang THẢ THẲNG XUỐNG (tư thế nghỉ): trọng lực gần như không tạo mô-men ở vai,
  nên giữ bằng PD không bù trọng lực vẫn an toàn.
- Một người cầm E-stop. Không ai đứng trong tầm với của tay.
- read_joints.py đã đọc được đủ 7 khớp trên interface này.

    python3 wiggle_j7.py --iface can0                # mặc định ±8°, chu kỳ 4 s, 2 chu kỳ
    python3 wiggle_j7.py --iface can0 --amp-deg 15

Cơ chế an toàn trong code:
- Đọc tư thế hiện tại trước khi enable; lệnh đầu tiên = đúng tư thế đó.
- Tăng gain từ 0 lên trong 1 s (soft start).
- Dừng ngay nếu một khớp lệch khỏi lệnh > --max-err-deg hoặc motor ngừng phản hồi.
- Kết thúc / Ctrl+C: về tư thế ban đầu, chuyển sang damping (kp=0) 1 s, rồi mới disable.
"""
import argparse
import math
import sys
import time

import openarm_can as oa
from openarm_common import (KP, KD, MECH_LIM_DEG, make_openarm, read_positions,
                            link_ok, fmt_deg)


class Abort(Exception):
    pass


def send(arm, q_cmd, kp_scale=1.0, damping_only=False):
    params = []
    for i in range(7):
        kp = 0.0 if damping_only else KP[i] * kp_scale
        params.append(oa.MITParam(kp, KD[i], q_cmd[i], 0.0, 0.0))
    arm.get_arm().mit_control_all(params)
    arm.recv_all(1000)
    return [m.get_position() for m in arm.get_arm().get_motors()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", required=True)
    ap.add_argument("--amp-deg", type=float, default=8.0)
    ap.add_argument("--period", type=float, default=4.0)
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--hz", type=float, default=100.0)
    ap.add_argument("--max-err-deg", type=float, default=12.0)
    ap.add_argument("--yes", action="store_true", help="bỏ bước hỏi xác nhận")
    args = ap.parse_args()
    if not (0 < args.amp_deg <= 30):
        sys.exit("amp-deg phải trong (0, 30]")

    arm = make_openarm(args.iface, with_gripper=False)

    # 1) Đọc tư thế khi motor còn tắt
    q0 = read_positions(arm, with_gripper=False)
    time.sleep(0.05)
    q0 = read_positions(arm, with_gripper=False)
    ok = link_ok(arm)
    if any(not math.isfinite(v) for v in q0) or (ok is not None and not all(ok)):
        sys.exit(f"Không đọc được đủ 7 khớp: {fmt_deg(q0)}  link={ok}. Chạy read_joints.py trước.")
    print("Tư thế hiện tại:", fmt_deg(q0))

    for i, (lo, hi) in enumerate(MECH_LIM_DEG):
        d = math.degrees(q0[i])
        if not (lo - 5 <= d <= hi + 5):
            sys.exit(f"J{i+1} = {d:.1f}° nằm ngoài giới hạn [{lo},{hi}] -> zero có thể sai, dừng.")
    j7 = math.degrees(q0[6])
    if abs(j7) + args.amp_deg > 85:
        sys.exit(f"J7 đang ở {j7:.1f}°, cộng biên độ sẽ vượt ±85°. Giảm --amp-deg.")

    if not args.yes:
        ans = input("Tay đang thả thẳng xuống, E-stop trong tay? Gõ 'yes' để chạy: ")
        if ans.strip().lower() != "yes":
            sys.exit("Huỷ.")

    dt = 1.0 / args.hz
    max_err = math.radians(args.max_err_deg)
    amp = math.radians(args.amp_deg)
    q_cmd = list(q0)

    def check(q_meas, q_cmd):
        for i in range(7):
            if not math.isfinite(q_meas[i]):
                raise Abort(f"J{i+1} không có dữ liệu")
            if abs(q_meas[i] - q_cmd[i]) > max_err:
                raise Abort(f"J{i+1} lệch {math.degrees(q_meas[i]-q_cmd[i]):.1f}° so với lệnh")
        ok = link_ok(arm, max_age_s=0.1)
        if ok is not None and not all(ok):
            raise Abort("mất phản hồi: " + ",".join(f"J{i+1}" for i, v in enumerate(ok) if not v))

    arm.enable_all()
    time.sleep(0.05)
    arm.recv_all(2000)
    q_meas = list(q0)
    try:
        # 2) Soft start: gain 0 -> 1 trong 1 s, lệnh = tư thế ban đầu
        t0 = time.perf_counter()
        while (s := (time.perf_counter() - t0) / 1.0) < 1.0:
            q_meas = send(arm, q_cmd, kp_scale=s)
            check(q_meas, q_cmd)
            time.sleep(dt)

        # 3) Lắc J7, biên độ tăng/giảm dần ở chu kỳ đầu/cuối
        total = args.period * args.cycles
        t0 = time.perf_counter()
        next_print = 0.0
        while (t := time.perf_counter() - t0) < total:
            env = min(1.0, t / args.period, (total - t) / args.period)
            q_cmd[6] = q0[6] + amp * env * math.sin(2 * math.pi * t / args.period)
            q_meas = send(arm, q_cmd)
            check(q_meas, q_cmd)
            if t >= next_print:
                print(f"{t:5.2f}s  J7 lệnh {math.degrees(q_cmd[6]):6.1f}°  đo {math.degrees(q_meas[6]):6.1f}°")
                next_print += 0.25
            time.sleep(dt)
        q_cmd = list(q0)
        for _ in range(int(0.5 * args.hz)):
            q_meas = send(arm, q_cmd)
            time.sleep(dt)
        print("Hoàn thành.")
    except Abort as e:
        print(f"!! DỪNG KHẨN: {e}")
    except KeyboardInterrupt:
        print("\nCtrl+C")
    finally:
        # 4) Damping 1 s rồi disable (tay đang thả xuống nên không rơi)
        try:
            for _ in range(int(1.0 * args.hz)):
                send(arm, q_meas, damping_only=True)
                time.sleep(dt)
        finally:
            arm.disable_all()
            arm.recv_all(2000)
            print("Đã disable. Tư thế cuối:", fmt_deg(read_positions(arm, with_gripper=False)))


if __name__ == "__main__":
    main()
