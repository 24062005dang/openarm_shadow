#!/usr/bin/env python3
"""Bước 1: chỉ ĐỌC góc 7 khớp + gripper. Không gửi lệnh mô-men nào.

    python3 read_joints.py --iface can0          # tay phải (theo quy ước nhóm)
    python3 read_joints.py --iface can1          # tay trái
    python3 read_joints.py --iface can0 --enable # nếu không đọc được khi motor đang tắt

Dùng để: xác nhận kết nối, xác định can0/can1 là tay nào (cầm tay khẽ xoay J7),
kiểm tra chiều dương từng khớp và so với tư thế zero.
--enable: bật motor nhưng KHÔNG gửi lệnh MIT nào -> motor vẫn mềm (giống openarm-can-cli monitor).
"""
import argparse
import math
import time

from openarm_common import make_openarm, read_positions, link_ok, fmt_deg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="can0")
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--enable", action="store_true")
    ap.add_argument("--no-gripper", action="store_true")
    args = ap.parse_args()
    with_grip = not args.no_gripper

    arm = make_openarm(args.iface, with_gripper=with_grip)
    if args.enable:
        arm.enable_all()
        time.sleep(0.1)
        arm.recv_all(2000)

    print(f"Đọc {args.iface} ở {args.hz} Hz. Ctrl+C để dừng.")
    t0 = time.time()
    warned = False
    try:
        while True:
            q = read_positions(arm, with_gripper=with_grip)
            ok = link_ok(arm)
            tag = "" if ok is None else ("" if all(ok) else
                  "  [mất phản hồi: " + ",".join(f"J{i+1}" for i, v in enumerate(ok) if not v) + "]")
            print(f"{time.time()-t0:6.1f}s  {fmt_deg(q)}{tag}")
            dead = (ok is not None and not any(ok)) or all(
                (not math.isfinite(v)) for v in q)
            if dead and time.time() - t0 > 1.5 and not warned:
                print("!! Không motor nào trả lời. Kiểm tra: nguồn robot, E-stop, "
                      "`candump can0`, đúng interface; hoặc chạy lại với --enable.")
                warned = True
            time.sleep(1.0 / args.hz)
    except KeyboardInterrupt:
        pass
    finally:
        if args.enable:
            arm.disable_all()
            arm.recv_all(2000)
        print("\nDừng.")


if __name__ == "__main__":
    main()
