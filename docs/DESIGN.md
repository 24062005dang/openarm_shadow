# Thiết kế

## Khung toạ độ
- Robot: khung world của URDF OpenArm v1.0 — x trước, y trái, z lên (kiểm tra bằng FK: J4 = +90° đưa cẳng tay ra +x,
  tay phải ở phía −y). Ở q = 0 hai tay thả xuôi.
- Người: khung thân dựng từ vai trái, vai phải, giữa hông (MakeFrame): cùng quy ước x trước, y trái, z lên.
  Thiếu hông (ngồi sau bàn) thì lấy hướng "lên" của camera.
- Vì hai khung cùng quy ước, hướng đoạn tay người đưa thẳng vào retarget, không cần hiệu chuẩn chiều dài.

## Lớp Arm (openarm_shadow/arm.py)
Mỗi tay robot là một `Arm(side)`; trái và phải dùng chung một lớp, `ShadowPipeline.arms` là `{side: Arm}`. `Arm` giữ
động học, bộ giải retarget, bộ lọc khớp, Kalman / EMA điểm mốc, kẹp, trạng thái hiệu chuẩn tay và các biến theo khung.
Các thuộc tính cũ (`pipe.held["right"]`, `pipe.rt["left"]`...) vẫn dùng được: chúng là view trỏ vào `Arm`.

Quan hệ trái / phải (đọc từ URDF, bảng trong `ROS inference.md` và MJCF v1; `tests/test_arm.py` kiểm tra ba nguồn):

| Khớp | Giới hạn trái so với phải | Trục khớp | Dấu khi phản chiếu tư thế (`MIRROR_SIGNS`) |
| --- | --- | --- | --- |
| J1, J2 | đảo (trái = −phải): J1 −200..80° / −80..200°, J2 −190..10° / −10..190° | giống nhau | −1 |
| J3, J5, J6 | đối xứng quanh 0 (±90°, ±90°, ±45°), giống nhau | giống nhau | −1 |
| J4 | giống nhau (0..140°) | giống nhau | +1 |
| J7 | đối xứng (±90°) | tay trái ngược tay phải (`0 -1 0`) | −1 |

Chỉ J1, J2 có giới hạn ngược nhau, nhưng phản chiếu tư thế (tay trái = ảnh gương tay phải) đảo dấu mọi khớp trừ J4:
hướng mọi link của tay trái trùng khít ảnh gương của tay phải (sai lệch 0°). Chỉ đảo J1, J2 thì sai tới ~157° khi cả 7
khớp cử động. Vị trí khớp lệch tới ~6 cm vì URDF không lật các độ lệch cơ khí giữa link; retarget chỉ dùng hướng nên không
ảnh hưởng. Chế độ `mapping.mode: mirror` hiện phản chiếu quan sát rồi giải riêng cho từng tay (kết quả khớp
`MIRROR_SIGNS * góc tay phải`, có test).

## Retarget (openarm_shadow/retarget.py)
OpenArm v1.0 có 7 khớp quay, hai trục liên tiếp luôn vuông góc (test `test_consecutive_axes_perpendicular`),
ở q = 0 trục J3 và J5 chạy dọc cánh tay trên và cẳng tay. Nên:

1. `align_axis(3)`: giải (J1, J2) để trục J3 trùng u — bài toán con SP2.
2. `align_axis(5)`: giải (J3, J4) để trục J5 trùng l — SP2. Tay gần thẳng: J3 không xác định → giữ J3 cũ, J4 bằng SP1.
3. `align_axis(7)` rồi SP1: giải (J5, J6) và J7 để khung link7 trùng `H · R_offset`.
   `R_offset` = phép quay nối hướng bàn tay người ở tư thế trung tính với hướng link7 ở q = 0 (phím `c` để đo lại).
4. Với mỗi bước: tối đa 2 nghiệm, chọn nghiệm trong giới hạn khớp và gần tư thế trước nhất (tránh nhảy nghiệm).

Đây là cách chia khớp thiết kế cho OpenArm dựa trên ý tưởng SEW-Mimic (căn hướng chi, SP1/SP2 dạng đóng).
Phần thuật toán chi tiết (mục IV) của bài báo chưa được đối chiếu, nên có thể khác cách tác giả làm.
Định nghĩa SP1/SP2 theo mục III-C của bài và ik-geo (Elias & Wen); cài đặt trong `geometry.py` tự viết.

## Tham khảo
- SEW-Mimic, arXiv 2602.01632
- Vision-Based Hand Shadowing, arXiv 2603.11383 (EMA hai tầng, kẹp có dự phòng, xem trước bằng mô phỏng, chế độ offline)
- Im-ma/Teleoperation-Arm (Marionette): lọc theo khớp, độ tin cậy từng khớp, dead-man, ghép bàn tay theo cổ tay
- YashVG/kinetic-arm-lab: ly hợp đặt lại gốc, dừng khi mất tracking
- perfanalytics/pose2sim: dùng làm ground truth khi đánh giá (2 camera, offline)
- rpiRobotics/ik-geo: cài đặt tham chiếu các bài toán con
- Dữ liệu động học: enactic/openarm_description commit 14ff67b (Apache-2.0), file `assets/robot/openarm_v1.0/urdf/example/v1.urdf`

## Hướng phát triển
- Chạy MediaPipe trong trình duyệt (GPU qua WebGL) nếu CPU laptop không đủ fps.
- Ghép 2 camera (điện thoại + laptop) để giảm sai số độ sâu.
- Kiểm tra va chạm tay–thân, bù trọng lực đã kiểm chứng, kẹp.
- Đánh giá: sai số góc so với Pose2Sim, độ trễ, tỉ lệ khung mất tay, độ giật (jerk).
