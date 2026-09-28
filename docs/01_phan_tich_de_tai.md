# Phân tích đề tài: teleop OpenArm bằng camera 2D (shadowing)

## Bài toán

Camera 2D (webcam laptop hoặc điện thoại) đặt **nhìn vào người**, không gắn trên đầu. Mô hình ước lượng tư thế vẽ bộ khớp
lên ảnh cánh tay; khi người cử động, hệ tính hướng/góc từng khớp rồi gửi sang OpenArm v1.0 thật qua CAN-FD.

Đây là hướng "vision-based shadowing" đã có nhiều người làm, nên **khả thi**. Phần học máy chỉ là mô hình ước lượng
tư thế có sẵn (MediaPipe); phần ánh xạ sang khớp robot là hình học, **không cần train, không cần dataset**.

## Độ khó từng khớp

| Khớp OpenArm | Lấy từ | Độ khó |
| --- | --- | --- |
| J1, J2 (vai nâng trước / dang ngang) | vector vai → khuỷu trong khung thân (2 vai + 2 hông) | Dễ |
| J3 (xoay cánh tay) | mặt phẳng vai – khuỷu – cổ tay | Trung bình; không xác định khi tay duỗi thẳng → giữ nguyên khi góc khuỷu nhỏ |
| J4 (khuỷu) | góc giữa cánh tay và cẳng tay | Dễ |
| J5 (xoay cẳng tay) | hướng lòng bàn tay (Hand Landmarker) so với cẳng tay | Khó, nhiễu |
| J6, J7 (cổ tay) | khung bàn tay so với cẳng tay | Trung bình – khó |
| Kẹp | khoảng cách ngón cái – ngón trỏ | Dễ |

## Rủi ro chính

1. **Một camera mù độ sâu.** Tay đưa về phía camera là trường hợp sai nhiều nhất. Giảm bằng cách đặt camera chéo ~45°,
   hoặc ghép 2 camera (điện thoại + laptop) để triangulate.
2. **Máy không có GPU NVIDIA.** Chỉ MediaPipe (CPU, ~30 fps) chạy thời gian thực được. WHAM/HaMeR (HumanPlus) cần GPU mạnh.
3. **Độ chính xác.** Kỳ vọng sai số góc cỡ 10–20° với một camera → hợp động tác biểu diễn / Thái Cực, chưa hợp gắp chính xác.
4. **Tay bị che hoặc ra khỏi khung.** Bài Hand Shadowing cho thấy đây là rủi ro lớn nhất ngoài đời thực
   (86,7 % trong phòng thí nghiệm, 9,3 % ngoài siêu thị). Hệ phải giữ nguyên lệnh khi mất tracking.
5. **Hai tay va nhau.** Khi gập cẳng tay trước ngực, J2 phải mở ra; cần kiểm tra khoảng cách tay – tay.

## Hai cách tính góc

1. **Giải tích / căn hướng.** Đổi các vector xương sang khung thân người rồi tách thành góc theo đúng thứ tự trục của
   OpenArm (giống HumanPlus copy góc Euler, Marionette ánh xạ góc-với-góc). Bản chặt chẽ của cách này là **SEW-Mimic**:
   căn hướng cánh tay trên, cẳng tay, bàn tay bằng các bài toán con SP1/SP2 có lời giải đóng. **Repo này dùng cách này.**
2. **IK trên model robot** (Pinocchio/pink): mục tiêu = vị trí khuỷu, cổ tay (co giãn theo chiều dài tay robot) + hướng
   bàn tay; tôn trọng giới hạn khớp. Chính xác vị trí hơn nhưng chậm hơn, có thể kỳ dị, và khuỷu tay 7 khớp dễ tự trôi
   nếu chỉ bám bàn tay.

Chỉ căn hướng nên tay người dài hay ngắn không ảnh hưởng; đổi lại, vị trí kẹp robot không trùng vị trí bàn tay người.

## Chỉ số đánh giá đề xuất

- Sai số góc từng khớp so với Pose2Sim (2 camera, offline) làm ground truth.
- Độ trễ đầu-cuối (ms), tỉ lệ khung mất tay, độ giật (jerk) quỹ đạo khớp.
- Nếu có tác vụ gắp: tỉ lệ thành công, thời gian hoàn thành so với leader arm.

## Phương án trước đó (đã thay)

Phương án điện thoại gắn trên đầu được phân tích trong [`07_phuong_an_dien_thoai_gan_dau.md`](07_phuong_an_dien_thoai_gan_dau.md).
Nhóm chuyển sang camera đặt ngoài vì từ đầu không nhìn thấy vai/khuỷu và phải có VIO để bù chuyển động đầu.
