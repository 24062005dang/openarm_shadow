# Hai bài báo: điểm mới và nên đưa gì vào bài toán

Chỉ dựa vào các phần đã đối chiếu được với bài gốc: tóm tắt và mục I–III của SEW-Mimic, tóm tắt và mục I–IV của
Hand Shadowing (bản dịch: [`research/dich_sew_mimic.md`](research/dich_sew_mimic.md),
[`research/dich_hand_shadowing.md`](research/dich_hand_shadowing.md)).

Hai bài bổ sung cho nhau: **SEW-Mimic làm lõi ánh xạ tay**, **Hand Shadowing cho các mẹo** để hệ chạy ổn định.

## SEW-Mimic (arXiv 2602.01632): nên làm lõi

### Có gì mới

1. **Chỉ so hướng, không so vị trí.** Robot không phải đưa bàn tay tới đúng chỗ bàn tay người, mà làm cho hướng
   cánh tay trên, cẳng tay và bàn tay trùng với người. Không phụ thuộc kích thước, không cần hiệu chuẩn chiều dài.
2. **Giải bằng công thức, không lặp.** Quy về bài toán con SP1, SP2 có lời giải đóng (mục III-C). Không Jacobian,
   không kỳ dị, ~3 kHz trên CPU → hợp laptop không GPU.
3. **Điều khiển khuỷu trực tiếp.** Tay 7 khớp như OpenArm: nếu chỉ bám bàn tay thì bậc thứ 7 cho khuỷu tự trôi.
   SEW-Mimic đưa khuỷu người vào bài toán nên robot giữ đúng dáng tay.
4. **Có bộ lọc chống hai tay va nhau.**

### Vì sao hợp OpenArm v1.0

- Trục tay phải ở q = 0: −y, −x, −z, −y, −z, +x, −y → hai trục liên tiếp vuông góc, đúng điều kiện của bài.
- Ở q = 0 trục J3 chạy dọc cánh tay trên, trục J5 dọc cẳng tay. Nên: hướng cánh tay trên → J1, J2; hướng cẳng tay → J3, J4;
  hướng bàn tay → J5–J7. Cách chia này **do nhóm suy ra**, chưa đối chiếu với mục IV của bài.

### Đã đưa vào repo

- Khung thân người (MakeFrame, Nhận xét 1 mục III-B): `geometry.make_frame`.
- SP1/SP2 tự cài: `geometry.sp1`, `sp2`; tham khảo [ik-geo](https://github.com/rpiRobotics/ik-geo) của Elias & Wen.
- Retarget: `retarget.py`. Kiểm chứng thử ngược 500 tư thế ngẫu nhiên, sai lệch ~1e-12°.
- Kiểm tra khoảng cách tay – tay bằng capsule: `safety.py` (bản đơn giản, không phải bộ lọc của bài).

### Lưu ý

- Robot chỉ tốt bằng điểm mốc đầu vào (bài tự nói). Nhiễu độ sâu của một webcam truyền thẳng sang góc khớp → vẫn phải lọc.
- Chứng minh tối ưu không tính giới hạn khớp. OpenArm J2 chỉ khép ~10°, J6 ±45° → kẹp giới hạn sẽ lệch tối ưu.
- Vị trí kẹp robot không trùng vị trí tay người: hợp biểu diễn/Thái Cực, chưa hợp gắp chính xác.

## Vision-Based Hand Shadowing (arXiv 2603.11383): lấy mẹo

### Có gì mới

Chủ yếu là **tích hợp và đánh giá thực tế**: pipeline chỉ CPU, không cần dữ liệu huấn luyện, trên tay robot giá rẻ (SO-ARM101).

- 86,7 % ± 4,2 % thành công trên bài gắp–đặt có cấu trúc.
- **Kết quả tiêu cực đáng giá:** ngoài đời (siêu thị, hiệu thuốc) chỉ còn 9,3 % vì tay bị che. WiLoR chỉ tăng tỉ lệ phát hiện tay 8 %.
- Bài học: rủi ro lớn nhất là **tay bị che / ra khỏi khung**, không phải IK.

### Nên lấy

| Mẹo | Trong repo |
| --- | --- |
| EMA điểm mốc α = 0,8 và EMA góc khớp α = 0,5 (giảm jerk 57–68 %) | EMA điểm mốc 0,8 trong `pipeline.py` (`filter.landmark_ema_alpha`); góc khớp dùng One Euro thay EMA |
| Bỏ khung khi < 50 % điểm hợp lệ | Độ tin cậy từng đoạn tay, giữ khớp khi thấp (`filters.JointFilter`) |
| Kẹp 3 mức dự phòng (đầu ngón → khớp ngón → giữ giá trị cũ) | Kẹp từ khoảng cách cái–trỏ, giữ giá trị cũ khi mất tay; kẹp robot đang tắt mặc định |
| Loại điểm đích dưới mặt bàn (z < 0,05 m) | Chưa có: mới kiểm tra tay – tay (xem SAFETY.md) |
| Xem trước trong mô phỏng rồi mới chạy thật | `--robot sim` là mặc định; MuJoCo v1 của nhóm dùng song song |
| Chế độ offline: quay trước, sinh quỹ đạo sau | `scripts/offline_retarget.py`, `replay_npz.py` |
| Chỉ số: sai số IK (mm), jerk, tỉ lệ phát hiện tay | Dùng cho báo cáo; ground truth bằng Pose2Sim |

### Không nên lấy

- Camera RGB-D gắn kính.
- IK chỉ bám vị trí bàn tay (khuỷu 7 khớp tự trôi).

## Ghép lại

1. Ánh xạ tay: SEW-Mimic.
2. Tín hiệu và kẹp: mẹo của Hand Shadowing + lọc từng khớp của Marionette.
3. An toàn: giới hạn khớp, vùng làm việc, kiểm tra tay – tay, xem trước mô phỏng.
4. Đánh giá: chỉ số của Hand Shadowing, ground truth Pose2Sim.

Phần thuật toán chi tiết (SEW-Mimic mục IV) và thí nghiệm của cả hai bài chưa đọc được bản gốc; khi có PDF sẽ bổ sung.
