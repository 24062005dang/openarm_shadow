# openarm_shadow

Teleop OpenArm v1.0 bằng camera 2D: robot bắt chước dáng tay người (shadowing).

Repo gồm hai phần: **code** chạy được (mô phỏng + OpenArm thật qua CAN-FD) và **tài liệu** nhóm đã tổng hợp
(phân tích đề tài, thông số OpenArm v1.0, bring-up, phân tích 5 repo, bản dịch 2 bài báo, tài liệu tham khảo).
Bắt đầu từ [`docs/README.md`](docs/README.md).

```
Webcam / điện thoại ──► MediaPipe Pose + Hand ──► khung thân người ──► retarget kiểu SEW-Mimic ──► lọc từng khớp
                                                                                  │
                         OpenArm v1.0 (CAN-FD) ◄── robot backend ◄── SafetyGate ◄──┘
                         hoặc robot mô phỏng
```

| Khâu | File | Lấy ý tưởng từ |
| --- | --- | --- |
| Nhận diện người + bàn tay (CPU) | `openarm_shadow/perception.py` | MediaPipe Tasks; ghép bàn tay theo cổ tay như repo Marionette |
| Khung thân (x trước, y trái, z lên) | `geometry.make_frame` | MakeFrame trong SEW-Mimic |
| Ánh xạ sang 7 khớp | `openarm_shadow/retarget.py` | SEW-Mimic (arXiv 2602.01632): căn **hướng** cánh tay, cẳng tay, bàn tay bằng bài toán con SP1/SP2 |
| Lọc | `openarm_shadow/filters.py` | One Euro + vùng chết + bỏ bước nhảy (Marionette), EMA điểm mốc 0.8 (Hand Shadowing) |
| An toàn | `openarm_shadow/safety.py` | Ly hợp + dead-man + tăng tốc mềm (Marionette, kinetic-arm-lab), capsule chống hai tay va nhau (SEW-Mimic) |
| Robot thật | `openarm_shadow/robot/openarm_can_robot.py` | `openarm_can`, gain chính thức v1.0 |
| Động học OpenArm v1.0 | `openarm_shadow/kinematics.py` + `data/openarm_v10_arms.json` | URDF của `enactic/openarm_description` |

Toàn bộ code tự viết; không chép code từ các repo trên (phần lớn không có license).

Repo này chỉ cho **teleop thời gian thực bằng camera**. Bài múa Thái Cực (phát lại quỹ đạo mocap đã tính sẵn,
`taichi_player`) là một hướng riêng của nhóm, không nằm ở đây. Hai repo chỉ dùng chung robot, bring-up CAN và cách
lọc số đọc rác.

## Trạng thái

- Đã kiểm tra (trên máy không có camera/robot): động học, retarget (thử ngược 500 tư thế, sai lệch < 1e-10°),
  bộ lọc, SafetyGate, luồng chạy sim với dữ liệu người giả lập, backend CAN với `openarm_can` giả lập.
  `pytest` 27/27 đạt (gồm test lọc số đọc rác của backend CAN).
- **Chưa chạy** với webcam thật và **chưa chạy trên OpenArm thật**. Làm theo `docs/SAFETY.md` trước khi bật motor.
- Quy ước góc URDF ↔ góc motor đang đặt là đồng nhất. **Phải kiểm tra** (SAFETY.md bước 3).

## Cài đặt (Ubuntu 24.04, Python 3.12, không cần GPU)

```bash
cd openarm_shadow
python3 -m venv --system-site-packages .venv   # system-site để thấy python3-openarm-can cài bằng apt
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_models.sh                # model MediaPipe vào models/
python -m pytest -q                            # 27 test phải đạt
python scripts/check_kinematics.py             # in trục khớp, thử ngược retarget
```

Robot thật cần thêm `openarm_can` (xem `tools/bringup/`): `sudo apt install python3-openarm-can`.

## Chạy

```bash
# 1) Mô phỏng: chỉ camera + hình que robot. Luôn chạy bước này trước.
python scripts/shadow.py
python scripts/shadow.py --mode mirror          # đứng đối diện robot, như soi gương
python scripts/shadow.py --arms right           # chỉ điều khiển tay phải

# 2) Chế độ offline: video quay sẵn -> quỹ đạo (thử pipeline khi chưa có robot, thu demo cho IL)
python scripts/offline_retarget.py demo.mp4 -o demo.npz --show
python scripts/replay_npz.py demo.npz            # xem lại trên robot mô phỏng

# 3) OpenArm thật (sau khi làm xong docs/SAFETY.md)
python scripts/shadow.py --robot openarm --arms right
```

Phím khi chạy: `SPACE` engage / nhả (ly hợp) · `c` hiệu chuẩn hướng bàn tay (đứng tay thả xuôi, lòng bàn
tay hướng vào đùi) · `p` về tư thế nghỉ · `q`/`Esc` về tư thế nghỉ rồi thoát.

Dùng điện thoại làm camera: cài app phát luồng video (vd. DroidCam, IP Webcam) rồi `--source http://<ip>:<port>/video`.

## Cách ánh xạ (tóm tắt)

- Từ MediaPipe: vai **s**, khuỷu **e**, cổ tay **w** và hướng bàn tay **H** (cổ tay, gốc ngón trỏ, gốc ngón út),
  đổi sang khung thân dựng từ hai vai và giữa hông.
- `u = unit(e − s)`, `l = unit(w − e)`. Với OpenArm v1.0, ở q = 0 trục J3 chạy dọc cánh tay trên, trục J5 dọc cẳng tay:
  - J1, J2: làm trục J3 trùng `u` (SP2)
  - J3, J4: làm trục J5 trùng `l` (SP2); tay gần thẳng (< 12°) thì giữ J3
  - J5, J6, J7: khớp hướng bàn tay `H` (SP2 + SP1)
- Chỉ dùng hướng nên tay người dài hay ngắn không ảnh hưởng. Đổi lại, vị trí kẹp robot không trùng vị trí bàn tay người:
  hợp với động tác biểu diễn, chưa hợp gắp chính xác.
- Nhiều nghiệm → chọn nghiệm trong giới hạn khớp và gần tư thế trước nhất.

Chi tiết và nguồn: `docs/DESIGN.md`, `docs/05_diem_moi_2_bai_bao.md`.

## Cấu trúc

```
config/default.yaml          mọi tham số (ghi đè bằng --config file_của_bạn.yaml)
openarm_shadow/              thư viện
scripts/shadow.py            chạy teleop (sim / openarm)
scripts/offline_retarget.py  video -> .npz
scripts/replay_npz.py        phát .npz qua SafetyGate
scripts/check_kinematics.py  kiểm tra động học + retarget
scripts/extract_kinematics.py sinh lại data JSON từ URDF
tools/bringup/               script bật CAN, đọc khớp, lắc J7 (bring-up robot)
tests/                       pytest
docs/README.md               mục lục tài liệu + lộ trình
docs/01..07_*.md             phân tích đề tài, OpenArm v1.0, bring-up, dữ liệu, 2 bài báo, tham khảo
docs/SAFETY.md, DESIGN.md    checklist an toàn, thiết kế code
docs/research/               phân tích 5 repo, bản dịch SEW-Mimic và Hand Shadowing
```
