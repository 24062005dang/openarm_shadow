#!/usr/bin/env python3
"""Tạo ảnh bảng ChArUco để in (A4, 300 dpi) theo fusion.board trong config.

    python scripts/make_charuco_board.py -o charuco_a4.png
In ở tỉ lệ 100% (không "fit to page"), dán lên tấm phẳng cứng. Đo cạnh 1 ô vuông đen bằng thước
rồi sửa fusion.board.square_m (và marker_m theo cùng tỉ lệ) nếu máy in co giãn.
"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openarm_shadow.cameras.calibration import make_board
from openarm_shadow.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="charuco_a4.png")
    ap.add_argument("--config", action="append", default=None)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    b = load_config(args.config)["fusion"]["board"]
    board, _ = make_board(b)
    mm = lambda m: int(round(m * 1000 / 25.4 * args.dpi))
    w, h = mm(b["squares_x"] * b["square_m"]), mm(b["squares_y"] * b["square_m"])
    a4w, a4h = int(210 / 25.4 * args.dpi), int(297 / 25.4 * args.dpi)
    if w > a4w or h > a4h:
        raise SystemExit(f"Bảng {w}x{h}px lớn hơn A4 ở {args.dpi} dpi: giảm square_m")
    img = board.generateImage((w, h), marginSize=0)
    page = cv2.copyMakeBorder(img, (a4h - h) // 2, a4h - h - (a4h - h) // 2, (a4w - w) // 2,
                              a4w - w - (a4w - w) // 2, cv2.BORDER_CONSTANT, value=255)
    cv2.imwrite(args.out, page)
    print(f"Đã ghi {args.out}: A4 {args.dpi} dpi, {b['squares_x']}x{b['squares_y']} ô, "
          f"ô {b['square_m'] * 1000:.1f} mm, marker {b['marker_m'] * 1000:.1f} mm, {b['dictionary']}")


if __name__ == "__main__":
    main()
