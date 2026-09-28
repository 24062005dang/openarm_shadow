# Trước khi chạy trên OpenArm thật

Làm lần lượt, không bỏ bước. Luôn có một người cầm E-stop, không ai đứng trong tầm với của tay.

## 1. Bring-up CAN
Làm theo `tools/bringup/README.md` (setup_can.sh → read_joints.py → wiggle_j7.py). Phải đọc được đủ 7 khớp
mỗi tay và lắc J7 thành công.

## 2. Mô phỏng trước
`python scripts/shadow.py` với webcam. Kiểm tra:
- hình que robot đi đúng chiều khi bạn nâng tay ra trước (J1), dang ngang (J2), gập khuỷu (J4), xoay cẳng tay;
- không có khớp nào giật khi đứng yên;
- khi che tay hoặc bước ra khỏi khung, robot đứng yên (dòng trạng thái báo hold).

## 3. Quy ước góc URDF ↔ motor (bắt buộc)
Code tính góc theo URDF v1.0. Góc motor = `sign · góc URDF + offset` (config `robot.urdf_to_motor`, mặc định sign = 1, offset = 0).
Kiểm tra bằng `tools/bringup/read_joints.py` (motor tắt, cầm tay robot):

| Động tác trên robot | Góc URDF phải | Góc URDF trái |
| --- | --- | --- |
| Tay thả xuôi | tất cả ≈ 0 | tất cả ≈ 0 |
| Nâng tay ra trước (J1) | dương | âm |
| Dang tay ra ngoài (J2) | dương | âm |
| Gập khuỷu (J4) | dương | dương |

Nhóm đã ghi nhận trước đây: J1/J2 dương ở tay phải, âm ở tay trái khi nâng; J4 dương khi gập → khớp với URDF.
J3, J5, J6, J7: so chiều với `python scripts/check_kinematics.py` hoặc viewer MuJoCo v1. Khớp nào ngược thì đặt sign = -1,
khớp nào lệch 0 thì đặt offset.

## 4. Lần chạy thật đầu tiên
```bash
python scripts/shadow.py --robot openarm --arms right
```
- Tay robot thả xuôi trước khi gõ `yes`. Lúc bật, gain tăng dần trong 1 s, lệnh bắt đầu đúng tư thế đo được.
- Bấm SPACE để engage: tốc độ tăng dần trong 1.5 s, tối đa 45°/s ở vai.
- Làm chậm, biên độ nhỏ. Hạ `safety.max_vel_deg_s` nếu cần.
- Giới hạn khớp mềm mặc định = giới hạn LeRobot (J1 ±75°, J2 phải −9…90°): tay không giơ quá cao được. Nới dần sau.

## 5. Những gì CHƯA có
- Bù trọng lực tắt mặc định. Không bù, với kp = 70 tay giơ ngang có thể võng khoảng 8° (ước tính từ mô men
  trọng lực ~10 Nm ở vai theo URDF). Bật `robot.gravity_comp` (cần `pip install pin` và đường dẫn URDF) sau khi thử từng khớp.
- Kẹp tắt mặc định: gripper 1.0 có thể đang ở chế độ POS_FORCE, và chiều mở/đóng chưa đo.
- Chống va chạm chỉ kiểm tra tay–tay, chưa kiểm tra tay–thân/cột và tay–bàn.
- Mất phản hồi CAN > 0.1 s: vòng điều khiển dừng, motor chuyển giảm chấn rồi tắt. Tay đang giơ sẽ hạ xuống từ từ,
  không giữ. E-stop vẫn là lớp bảo vệ cuối.
