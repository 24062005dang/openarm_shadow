# Lệnh calib và chạy OpenArm

File này là nơi lưu lệnh chuẩn cho bộ phần cứng hiện tại. Khi camera, robot hoặc config thay đổi,
cập nhật lệnh tại đây.

## 1. Calib lại ba camera

Chạy sau mỗi lần di chuyển camera hoặc thay đổi góc màn hình laptop:

```bash
cd /home/d/Downloads/Openarm_chaylai

.venv/bin/python scripts/calibrate_cameras.py \
  --config config/local_3cam.yaml \
  --samples 20
```

- Bảng: ChArUco `7x5`, ô `40 mm`, marker `30 mm`, `DICT_5X5_1000`.
- Đưa bảng qua vùng gắp/thả; camera front và từng camera phụ phải cùng thấy bảng.
- Không dùng `--redo-intrinsics` nếu chỉ di chuyển D435/D455.
- Nên calib lại nếu sai số chiếu lại trên `3 px`; không dùng kết quả trên `6 px`.

## 2. Chạy robot thật: hai tay, cổ tay và kẹp

```bash
cd /home/d/Downloads/Openarm_chaylai

.venv/bin/python scripts/shadow.py \
  --source multi \
  --robot openarm \
  --config config/d455_wrist_real.yaml \
  --config config/gripper_real.yaml \
  --config config/local_both_full.yaml \
  --config config/local_3cam.yaml \
  --record run_grasp_drop.npz
```

Sau khi gõ `yes`, không cần bấm SPACE để bắt đầu. Thả hai tay xuống, xòe bàn tay, lòng bàn tay
hướng về camera và giữ `READY` liên tục 2 giây. Robot tự engage một lần và ramp trong 3 giây.

- `SPACE`: nhả/dừng robot; sau khi nhả sẽ không tự engage lại.
- `p`: về tư thế nghỉ nhưng tiếp tục chương trình.
- `q` hoặc `Esc`: về tư thế nghỉ và thoát.
- Luôn giữ E-stop trong tầm tay.

## 3. Thu dữ liệu camera thô để phân tích offline

Lệnh này không enable motor và không làm robot chuyển động:

```bash
cd /home/d/Downloads/Openarm_chaylai

.venv/bin/python scripts/record_multicam_raw.py \
  --config config/local_3cam.yaml \
  --out data/2026-10-03_raw_bimanual_take2 \
  --seconds 60 \
  --countdown 10
```

Trong 60 giây ghi: đứng nghỉ hai tay xuôi; đưa từng tay tới nhiều vị trí trong vùng làm việc;
xoay hai cổ tay và đóng/mở ngón; cầm một vật rồi chuyển từ tay này sang tay kia; lặp lại
một lần chậm và một lần nhanh. Giữ cả đầu, vai, khuỷu và hai bàn tay trong vùng nhìn chung.
