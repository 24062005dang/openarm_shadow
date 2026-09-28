# Bring-up OpenArm v1.0 trên Ubuntu 24.04 cài thẳng (SavvyCAN-FD-X2)

Script nằm ở [`tools/bringup/`](../tools/bringup/). Các lệnh dưới chạy từ thư mục đó (`cd tools/bringup`).
Đã thử logic bằng `openarm_can` giả lập, **chưa chạy trên robot thật**.

Thứ tự: cài đặt → bật CAN → đọc khớp → chuyển động đầu tiên. Chưa làm xong bước trước thì không sang bước sau.

## 0. An toàn
- Luôn có một người cầm E-stop khi robot có nguồn.
- Không chạy `set_zero` / `openarm-can-cli ... set_zero` trừ khi cả nhóm thống nhất hiệu chuẩn lại. Lệnh này ghi đè zero của motor.
- Tay phải thả thẳng xuống trước khi enable. Code hiện chưa có bù trọng lực.

## 1. Cài đặt (một lần)
```bash
sudo apt update
sudo apt install -y can-utils git python3-venv software-properties-common
sudo add-apt-repository -y ppa:openarm/main
sudo apt update
sudo apt install -y libopenarm-can-dev openarm-can-utils python3-openarm-can
python3 -c "import openarm_can; print('openarm_can OK')"
```
Nếu dùng venv (như README gốc của repo), tạo bằng `python3 -m venv --system-site-packages .venv` để venv vẫn thấy `openarm_can` cài từ apt.

## 2. Cắm adapter và bật CAN
1. Cắm SavvyCAN-FD-X2 vào laptop, rồi nối kênh 1 và kênh 2 vào hai tay robot.
2. Bật nguồn robot.
3. Chạy:
```bash
chmod +x setup_can.sh && ./setup_can.sh
candump -n 20 can0
```
- Trên Linux, adapter hiện ra là PEAK PCAN-USB Pro FD (`0c72:0011`, driver `peak_usb`). Driver này có sẵn trong kernel, không cần cài thêm.
- Phải chạy lại `setup_can.sh` mỗi lần cắm lại adapter hoặc khởi động lại máy.

## 3. Đọc khớp (chưa có lực)
```bash
python3 read_joints.py --iface can0
python3 read_joints.py --iface can1
```
- Xoay nhẹ cổ tay (J7) của từng tay để biết can0 là tay nào. Quy ước cũ của nhóm: can0 = phải, can1 = trái. Thứ tự này có thể đổi khi cắm lại USB.
- Đưa tay về tư thế zero rồi xem các góc có gần 0 không. Kiểm tra chiều dương của từng khớp so với quy ước nhóm đã ghi.
- Nếu không motor nào trả lời, chạy lại với `--enable`. Motor được bật nhưng không nhận lệnh MIT, nên vẫn mềm.

## 4. Chuyển động đầu tiên
```bash
python3 wiggle_j7.py --iface can0
```
Script giữ J1–J6 và lắc J7 ±8°. Nếu một khớp lệch quá 12° so với lệnh hoặc motor ngừng phản hồi, script tự dừng.

## Sự cố thường gặp
| Hiện tượng | Kiểm tra |
|---|---|
| Không có `can0` | `lsusb`, `sudo dmesg \| grep -i peak`, `sudo modprobe peak_usb` |
| `candump` không có frame | Nguồn robot và E-stop đã nhả chưa, dây CAN H/L, điện trở đầu cuối 120 Ω |
| `ip -details link show can0` báo BUS-OFF | Sai bitrate hoặc lỗi dây. Chạy lại `setup_can.sh` |
| Một khớp báo `---` | Motor đó không trả lời: kiểm tra ID, dây nối tiếp giữa các motor |

## API `openarm_can` đã dùng (commit f340d4b)

Python binding được đánh dấu "UNSTABLE API", có thể đổi giữa các bản. Các hàm repo dùng:

```python
import openarm_can as oa
arm = oa.OpenArm("can0", True)                         # True = CAN-FD
arm.init_arm_motors(motor_types, send_ids, recv_ids)   # 0x01..0x07 / 0x11..0x17
arm.init_gripper_motor(oa.MotorType.DM4310, 0x08, 0x18)
arm.set_callback_mode_all(oa.CallbackMode.STATE)
arm.refresh_all(); arm.recv_all(2000)                  # timeout µs
q = [m.get_position() for m in arm.get_arm().get_motors()]
arm.get_arm().mit_control_all([oa.MITParam(kp, kd, q, dq, tau), ...])
arm.get_arm().get_link_stats(i).seconds_since_response()
```

Cấu hình CAN bằng công cụ chính thức (thay cho `setup_can.sh`): `openarm-can-cli -i can0 can_configure`
(mặc định 1 Mbps / 5 Mbps FD, sample point 0,75).

## Sau bring-up

1. Kiểm tra quy ước góc URDF ↔ motor: [`SAFETY.md`](SAFETY.md) bước 3.
2. Chạy teleop ở chế độ mô phỏng, rồi mới `--robot openarm --arms right`.
3. Bù trọng lực (`robot.gravity_comp`, cần `pip install pin`) trước khi giơ tay cao khỏi tư thế nghỉ lâu.
