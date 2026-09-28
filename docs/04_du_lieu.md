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

## Chế độ offline (video → quỹ đạo)

- Quay video → `offline_retarget.py` → `.npz` → `replay_npz.py` (mô phỏng trước, robot thật chậm `--speed 0.5`).
- Dùng để thử pipeline khi chưa có robot, và để thu demo cho imitation learning.
- Múa Thái Cực từ mocap có sẵn là hướng riêng (`taichi_player`), không làm bằng repo này.

## Imitation learning sau này

Không có dataset công khai khớp OpenArm v1.0 + camera của nhóm. Dữ liệu IL = demo do nhóm tự teleop và ghi lại
(`scripts/shadow.py --record out.npz`, hoặc chuyển sang định dạng LeRobot nếu dùng `lerobot-record`).
Chế độ offline (video → quỹ đạo) cũng dùng để sinh demo khi teleop thời gian thực chưa ổn.

## License cần nhớ

- EgoDex: CC-BY-NC-ND. LAFAN1: CC BY-NC-ND 4.0. AMASS: nghiên cứu phi thương mại.
- Pose2Sim: BSD-3. MediaPipe: Apache-2.0. `openarm_description`: Apache-2.0.
