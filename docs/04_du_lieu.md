# Dữ liệu: cần gì, lấy ở đâu

## Teleop không cần dataset

Đầu vào của teleop là chuyển động của người thao tác, thu trực tiếp từ camera. MediaPipe là model đã train sẵn;
phần ánh xạ sang 7 khớp là hình học. Nên để **robot chạy được thì không cần tải dữ liệu nào**, chỉ cần model MediaPipe
(`bash scripts/download_models.sh`).

## Dữ liệu dùng để phát triển và đo sai số

| Mục đích | Nguồn | Ghi chú |
| --- | --- | --- |
| Thử pipeline khi chưa có robot | Video tự quay bằng webcam/điện thoại | `scripts/offline_retarget.py video.mp4 -o out.npz --show` |
| Ground truth góc khớp | Quay cùng lúc 2 camera (điện thoại + laptop) → Pose2Sim offline | 1 camera phía trước, 1 camera lệch 45°, ngang hông; ≥ 30 fps |
| Ground truth tay/cánh tay 3D (egocentric) | EgoDex (Apple, `github.com/apple/ml-egodex`) | Có pose vai, cánh tay, cẳng tay, 25 khớp bàn tay; test set 16 GB; CC-BY-NC-ND. Góc nhìn từ đầu, hợp phương án gắn đầu hơn |
| Dự phòng egocentric | HOT3D (Meta), MobileEgo Anywhere | |

## Nhánh Thái Cực / động tác biểu diễn

- Chế độ offline: quay một người tập → `offline_retarget.py` → `.npz` → `replay_npz.py` (mô phỏng, rồi robot thật chậm `--speed 0.5`).
- Tốt hơn nếu quay 2 camera và chạy Pose2Sim để có góc chính xác hơn.
- Nguồn motion có sẵn (định dạng G1, cần retarget sang OpenArm): KungfuAthleteBot (Apache-2.0/MIT, có 42 thức Thái Cực),
  CMU qua AMASS (12_04 Thái Cực; phi thương mại), LAFAN1 (CC BY-NC-ND), g1-moves. Chi tiết trong tài liệu project
  "motion-sources" của nhóm.
- Bài cơ bản gợi ý: Bát đoạn cẩm (động tác 1, 2, 3, 7 chỉ dùng tay, chậm), port de bras ballet (đối xứng, chậm).

## Imitation learning sau này

Không có dataset công khai khớp OpenArm v1.0 + camera của nhóm. Dữ liệu IL = demo do nhóm tự teleop và ghi lại
(`scripts/shadow.py --record out.npz`, hoặc chuyển sang định dạng LeRobot nếu dùng `lerobot-record`).
Chế độ offline (video → quỹ đạo) cũng dùng để sinh demo khi teleop thời gian thực chưa ổn.

## License cần nhớ

- EgoDex: CC-BY-NC-ND. LAFAN1: CC BY-NC-ND 4.0. AMASS: nghiên cứu phi thương mại.
- Pose2Sim: BSD-3. MediaPipe: Apache-2.0. `openarm_description`: Apache-2.0.
