# Tài liệu dự án: teleop OpenArm v1.0 bằng camera 2D

Mọi thứ nhóm và Claude đã tổng hợp, theo thứ tự nên đọc.

| # | File | Nội dung |
| --- | --- | --- |
| 1 | [`01_phan_tich_de_tai.md`](01_phan_tich_de_tai.md) | Bài toán, tính khả thi, độ khó từng khớp, rủi ro, hai cách tính góc, chỉ số đánh giá |
| 2 | [`02_openarm_v10.md`](02_openarm_v10.md) | Thông số OpenArm v1.0: trục khớp, giới hạn URDF vs motor, motor, CAN ID, gain, hiệu chuẩn |
| 3 | [`03_bringup_ubuntu.md`](03_bringup_ubuntu.md) | Bring-up trên Ubuntu 24.04 native qua SavvyCAN-FD-X2, API `openarm_can` |
| 4 | [`SAFETY.md`](SAFETY.md) | Checklist bắt buộc trước khi chạy trên robot thật |
| 5 | [`DESIGN.md`](DESIGN.md) | Thiết kế code: khung toạ độ, thuật toán retarget |
| 6 | [`04_du_lieu.md`](04_du_lieu.md) | Cần dữ liệu gì, nguồn nào, license |
| 7 | [`05_diem_moi_2_bai_bao.md`](05_diem_moi_2_bai_bao.md) | Điểm mới của SEW-Mimic và Hand Shadowing, đã đưa gì vào repo |
| 8 | [`06_tai_lieu_tham_khao.md`](06_tai_lieu_tham_khao.md) | Danh sách bài báo, repo, dữ liệu |
| 9 | [`07_phuong_an_dien_thoai_gan_dau.md`](07_phuong_an_dien_thoai_gan_dau.md) | Phương án cũ (điện thoại gắn đầu) và lý do đổi |

Tài liệu nghiên cứu gốc:

| File | Nội dung |
| --- | --- |
| [`research/phan_tich_5_repo.md`](research/phan_tich_5_repo.md) | Phân tích chi tiết 5 repo (openarm_teleoperation, robot-teleop, Marionette, kinetic-arm-lab, Pose2Sim) + pipeline đề xuất |
| [`research/dich_sew_mimic.md`](research/dich_sew_mimic.md) | Bản dịch SEW-Mimic: tóm tắt, mục I–III |
| [`research/dich_hand_shadowing.md`](research/dich_hand_shadowing.md) | Bản dịch Hand Shadowing: tóm tắt, mục I–IV |

> Hai bản dịch **chưa đủ**: SEW-Mimic từ mục IV, Hand Shadowing từ mục V trở đi chưa dịch vì chưa lấy được bản gốc đầy đủ.
> Các đoạn đánh dấu ★ OpenArm v1.0 là ghi chú của nhóm, không có trong bài gốc.

## Lộ trình

| Bước | Việc | Trạng thái |
| --- | --- | --- |
| 1 | Bring-up CAN, đọc khớp, lắc J7 | Có script; CAN đã chạy với taichi_player (WSL2), chưa chạy script này trên Ubuntu native |
| 2 | Kiểm tra quy ước góc URDF ↔ motor (SAFETY.md bước 3) | J1–J4 đã khớp (đo 25/09); J5–J7 chưa |
| 3 | Mô phỏng: `scripts/demo_sim.py` (không camera), rồi `scripts/shadow.py` với webcam thật | Xong: demo_sim và webcam thật đều chạy (28/09) |
| 4 | Dry-run (motor tắt) rồi chạy thật tay phải, J1–J4, `config/first_real.yaml` | Chưa làm |
| 5 | Thêm J5–J7, kẹp; bật bù trọng lực | Code có, tắt mặc định |
| 6 | Hai tay, chế độ gương | Code có |
| 7 | Đo sai số bằng Pose2Sim (2 camera), độ trễ, jerk | Chưa làm |
| 8 | Chế độ offline: video → `offline_retarget.py` → `replay_npz.py` | Code có |
| 9 | Thu demo cho imitation learning | Sau cùng |

## Quy tắc an toàn không đổi

- Luôn có người cầm E-stop; không ai đứng trong tầm với.
- Tay robot thả xuôi trước khi bật motor.
- Không chạy `set_zero` trừ khi cả nhóm quyết định (v1.0: `--robot-version v1`).
- Chưa kiểm tra quy ước góc URDF ↔ motor thì không chạy teleop trên robot thật.
