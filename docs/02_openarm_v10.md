# OpenArm v1.0: thông số dùng trong repo

Nguồn: `enactic/openarm_description` commit 14ff67b (Apache-2.0), file `assets/robot/openarm_v1.0/urdf/example/v1.urdf`;
`openarm_can` commit f340d4b; LeRobot `openarm_follower`. Dữ liệu động học đã trích ra `openarm_shadow/data/openarm_v10_arms.json`
(sinh lại bằng `scripts/extract_kinematics.py`).

## Khung và tư thế 0

- Khung world của URDF: x trước, y trái, z lên. Ở q = 0 hai tay **thả thẳng xuống**
  (FK tay phải: khuỷu cách vai ~0,22 m, cổ tay ~0,44 m phía dưới).
- Tên khớp `openarm_{right,left}_joint1..7`, kẹp `..._finger_joint1`.
- MJCF: `openarm_mujoco/v1/`.

## Trục khớp tay phải ở q = 0

| Khớp | Trục | Ý nghĩa |
| --- | --- | --- |
| J1 | −y | nâng tay ra trước |
| J2 | −x | dang tay ra ngoài |
| J3 | −z | xoay cánh tay trên (dọc cánh tay) |
| J4 | −y | gập khuỷu |
| J5 | −z | xoay cẳng tay (dọc cẳng tay) |
| J6 | +x | cổ tay |
| J7 | −y | cổ tay |

- Hai trục liên tiếp luôn vuông góc (test `test_consecutive_axes_perpendicular`) → hợp điều kiện của SEW-Mimic.
- Vai không phải khớp cầu lý tưởng: J2/J3 lệch nhau 3 cm theo x, J4 lệch 3,15 cm theo y. Cách căn hướng không bị ảnh hưởng;
  cách giải tích dựa vị trí chỉ gần đúng.
- Tay trái: J1 và J7 đảo trục. Dấu mirror phải → trái (J1..J7, theo nhánh gesture): `[−1, −1, −1, 1, −1, −1, −1]`.

## Giới hạn khớp (độ)

| Khớp | URDF phải | URDF trái | LeRobot / motor phải | LeRobot / motor trái |
| --- | --- | --- | --- | --- |
| J1 | [−80, 200] | [−200, 80] | ±75 | ±75 |
| J2 | [−10, 190] | [−190, 10] | [−9, 90] | [−90, 9] |
| J3 | ±90 | ±90 | ±85 | ±85 |
| J4 | [0, 140] | [0, 140] | [0, 135] | [0, 135] |
| J5 | ±90 | ±90 | ±85 | ±85 |
| J6 | ±45 | ±45 | ±40 | ±40 |
| J7 | ±90 | ±90 | ±80 | ±80 |
| Kẹp | | | [−65, 0] | [−65, 0] |

- Repo dùng giới hạn LeRobot làm giới hạn mềm mặc định (`config/default.yaml` → `safety`).
- J2 chỉ khép vào trong ~10° (người khép tay qua ngực được nhiều hơn) → sẽ bị kẹp giới hạn; J6 chỉ ±40–45°.
- Xacro của URDF cộng offset (J1 trái −120°, J2 ±90°), nên **quy ước góc URDF và góc motor có thể khác nhau**.
  Phải đo đối chiếu bằng `tools/bringup/read_joints.py` trước khi chạy thật (xem [`SAFETY.md`](SAFETY.md) bước 3).

## Motor và CAN

- Motor: DM8009 × 2 (J1, J2), DM4340 × 2 (J3, J4), DM4310 × 3 (J5–J7) + kẹp DM4310.
- CAN ID gửi 0x01–0x08, nhận 0x11–0x18. Mỗi tay một bus (repo mặc định: `can0` tay phải, `can1` tay trái — phải xác nhận).
- CAN-FD 1 Mbps / 5 Mbps.
- Gain chính thức v1.0: kp `70, 70, 70, 60, 10, 10, 10`, kd `2.75, 2.5, 2, 2, 0.7, 0.6, 0.5`.
  LeRobot mặc định kp 240, cao hơn nhiều.
- Mô-men trọng lực ở vai khi tay giơ ngang ~10,4 Nm (tính bằng pinocchio trên URDF) → với kp 70 tay võng ~8° nếu không bù.

## Hiệu chuẩn zero

Chỉ làm khi cả nhóm quyết định:

```bash
openarm-can-zero-position-calibration --canport can0 --arm-side right_arm --robot-version v1
```

Mặc định công cụ là v2, **phải truyền `--robot-version v1`**.

## LeRobot

`openarm_follower` dùng được cho v1.0 (cùng motor, CAN ID). Nhớ đặt `--robot.side` để có giới hạn khớp, nếu không
mặc định chỉ ±5°.
