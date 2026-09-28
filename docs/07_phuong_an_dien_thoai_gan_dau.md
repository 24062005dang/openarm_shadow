# Phương án cũ: điện thoại gắn trên đầu (đã thay)

Giữ lại để tham khảo. Nhóm đã chuyển sang camera đặt ngoài nhìn vào người ([`01_phan_tich_de_tai.md`](01_phan_tich_de_tai.md)).

## Bản chất

Teleop dựa trên thị giác egocentric: camera sau của điện thoại gắn trán nhìn xuống vùng tay, ước lượng tư thế tay/cánh tay,
ánh xạ sang 7 khớp + kẹp của OpenArm.

## Vì sao khó

1. **Tầm nhìn.** Từ đầu chỉ thấy bàn tay và cẳng tay; vai không bao giờ trong khung, khuỷu thường ngoài khung.
   MediaPipe Pose không dùng được cho chính người đeo. Phải đo cổ tay 6D + hướng cẳng tay, rồi suy ra vai
   (từ tư thế đầu + offset hiệu chuẩn, giả sử thân ngồi yên) và khuỷu (vai – cổ tay + chiều dài tay + hướng cẳng tay).
2. **Đầu chuyển động → khung camera chuyển động.** Bắt buộc dùng VIO (ARKit/ARCore) để đưa tư thế tay về khung thế giới cố định;
   nếu không, quay đầu = robot giật.
3. **Độ sâu đơn mắt.** MediaPipe chỉ cho độ sâu tương đối. Cần iPhone Pro có LiDAR, hoặc HaMeR/WiLoR, hoặc prior kích thước bàn tay.
4. **Tay ra khỏi khung khi giơ cao / dang ngang** (xung đột với động tác Thái Cực) → giữ lệnh cuối + cảnh báo.
5. **Dư bậc tự do.** Cổ tay 6D chỉ ràng 6 bậc → cần chọn góc xoay khuỷu (kiểu SEW-Mimic).
6. Trễ và rung, giới hạn khớp, chống va chạm hai tay, dead-man + E-stop, kẹp từ khoảng cách cái – trỏ.

## Pipeline đã đề xuất

Điện thoại (ARKit world tracking + hand 21 điểm + LiDAR, 30–60 Hz) → WebSocket/UDP → PC: đổi về khung thế giới → khung robot
(hiệu chuẩn một lần) → retarget (IK có trọng số cổ tay 6D + khuỷu, hoặc SEW-Mimic) → lọc + an toàn → `openarm_can` / LeRobot.
Baseline: leader–follower `openarm_teleop` / LeRobot `openarm_leader`.

## Dữ liệu và tài liệu riêng cho phương án này

EgoDex (có pose vai/cánh tay/cẳng tay + 25 khớp tay + pose camera ARKit), HOT3D, MobileEgo Anywhere, EgoForce,
Open-TeleVision, `xr_teleoperate`, `SpesRobotics/teleop`, `trzy/robot-arm`. Link ở [`06_tai_lieu_tham_khao.md`](06_tai_lieu_tham_khao.md).
