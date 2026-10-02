# Giao diện ROS 2 của OpenArm: Joint States & Teleop

Tài liệu này mô tả các topic ROS 2 mà backend OpenArm (`sim/server.py`, thông qua`sim/openarm_joint_bridge.py`) publish và subscribe: trạng thái khớp, lệnh khớpvà các topic điều khiển teleop.

## 1. Quy ước chung

### Quy ước góc khớp: URDF chính thức của OpenArm

Mọi góc khớp tay trên mọi topic, trong dataset và trên Web UI đều theo **URDF chínhthức của OpenArm**: **góc khớp = góc motor thật**. Không có khớp nào bị đổi dấu ởbất kỳ tầng nào.

Hai tay lắp đối xứng gương, nên một số khớp có giới hạn ngược nhau giữa trái vàphải, đúng như trong URDF. Ví dụ với J1 (vai pitch), `+q` đưa tay phải ra trướcnhưng đưa tay trái ra sau.

IK giải bằng URDF chính thức có thể gửi thẳng góc sang robot, không cần chuyển đổi.

### Đơn vị

| Loại            | Vị trí                                                              | Vận tốc | Effort            || ---------------- | --------------------------------------------------------------------- | --------- | ----------------- || Khớp tay J1..J7 | rad                                                                   | rad/s     | Nm (torque motor) || Gripper          | **m** (hành trình ngón kẹp, 0.0 = đóng, 0.043 = mở hết) | m/s       | Nm (torque motor) |

Gripper chỉ dùng radian ở tầng CAN: 0.0 m tương ứng 0.0 rad, 0.043 m tương ứng−1.20 rad. Việc quy đổi nằm trong `gripper_stroke_to_rad` / `gripper_rad_to_stroke`ở `sim/config.py`.

### Thứ tự và giới hạn 16 khớp

Thứ tự này được dùng cho `/openarm/joint_states`, `/openarm/joint_commands` vàdataset (`qpos`, `action`). Giới hạn lấy từ `JOINT_LIMITS` trong `sim/config.py`.Backend tự clamp mọi lệnh về trong giới hạn.

| Index | Tên (`name`)   | Motor ID | CAN  | Giới hạn          | Đơn vị || ----- | ----------------- | -------- | ---- | ------------------- | --------- || 0     | `left_j1`       | 1        | can1 | −3.4907 .. 1.3963  | rad       || 1     | `left_j2`       | 2        | can1 | −3.3161 .. 0.17453 | rad       || 2     | `left_j3`       | 3        | can1 | −1.5708 .. 1.5708  | rad       || 3     | `left_j4`       | 4        | can1 | 0.0 .. 2.4435       | rad       || 4     | `left_j5`       | 5        | can1 | −1.5708 .. 1.5708  | rad       || 5     | `left_j6`       | 6        | can1 | −0.7854 .. 0.7854  | rad       || 6     | `left_j7`       | 7        | can1 | −1.5708 .. 1.5708  | rad       || 7     | `left_gripper`  | 8        | can1 | 0.0 .. 0.043        | m         || 8     | `right_j1`      | 9        | can0 | −1.3963 .. 3.4907  | rad       || 9     | `right_j2`      | 10       | can0 | −0.17453 .. 3.3161 | rad       || 10    | `right_j3`      | 11       | can0 | −1.5708 .. 1.5708  | rad       || 11    | `right_j4`      | 12       | can0 | 0.0 .. 2.4435       | rad       || 12    | `right_j5`      | 13       | can0 | −1.5708 .. 1.5708  | rad       || 13    | `right_j6`      | 14       | can0 | −0.7854 .. 0.7854  | rad       || 14    | `right_j7`      | 15       | can0 | −1.5708 .. 1.5708  | rad       || 15    | `right_gripper` | 16       | can0 | 0.0 .. 0.043        | m         |

Tên khớp tương ứng trong URDF chính thức: `left_jN` ↔ `openarm_left_jointN`,`left_gripper` ↔ `openarm_left_finger_joint1`, tương tự cho tay phải. File`sim/web/assets/openarm_v1/.../urdf/openarm_v1_codebase.urdf` là URDF chính thức đãđổi sang tên khớp ở trên, dùng được trực tiếp với `robot_state_publisher`.

## 2. Topic publish (robot → ROS 2)

### `/openarm/joint_states`: trạng thái thực tế

| Thuộc tính | Giá trị                                                                                               || ------------ | ------------------------------------------------------------------------------------------------------- || Kiểu        | `sensor_msgs/msg/JointState`                                                                          || Tần số     | 100 Hz                                                                                                  || QoS          | `RELIABLE`, `KEEP_LAST` depth 5 (subscriber `RELIABLE` hoặc `BEST_EFFORT` đều nhận được) || `name`     | 16 tên theo bảng trên                                                                                || `position` | vị trí đọc từ encoder (rad / m)                                                                    || `velocity` | vận tốc (rad/s / m/s)                                                                                 || `effort`   | torque motor (Nm)                                                                                       |

Đây là nguồn `observations/qpos`, `qvel` và `effort` của dataset ACT.

### `/openarm/joint_commands`: mục tiêu đang áp dụng

| Thuộc tính | Giá trị                                                                          || ------------ | ---------------------------------------------------------------------------------- || Kiểu        | `sensor_msgs/msg/JointState`                                                     || Tần số     | 100 Hz                                                                             || QoS          | `BEST_EFFORT`, `KEEP_LAST` depth 5                                             || `position` | 16 vị trí mục tiêu (rad / m) sau khi đã gộp lệnh và clamp theo giới hạn |

Đây là nguồn `action` của dataset ACT. Topic này là mục tiêu cuối, chưa qua nội suy.Backend di chuyển motor tới mục tiêu bằng bộ nội suy 400 Hz, giới hạn vận tốc mặcđịnh 0.25 rad/s cho khớp tay. Vì vậy `joint_states` sẽ theo kịp `joint_commands`sau một khoảng thời gian.

## 3. Topic teleop (ROS 2 → robot)

Bridge subscribe với QoS `BEST_EFFORT` depth 10. Phía teleop có thể publish với`BEST_EFFORT` (khuyên dùng để giảm độ trễ) hoặc `RELIABLE`.

### `/openarm/teleop/joint_commands`: 14 khớp tay

| Thuộc tính | Giá trị                                                                              || ------------ | -------------------------------------------------------------------------------------- || Kiểu        | `std_msgs/msg/Float64MultiArray`                                                     || `data`     | **đúng 14 giá trị** (rad): `left_j1..left_j7`, rồi `right_j1..right_j7` || `layout`   | không dùng (để trống)                                                             |

- Không có gripper trong topic này. Gripper đi qua hai topic riêng bên dưới.- Message có số phần tử khác 14 sẽ bị từ chối, bridge log cảnh báo  `Rejected teleop arm command`.- Muốn giữ một tay đứng yên thì phải gửi vị trí hiện tại của tay đó. Gửi `0.0` sẽ  kéo tay về vị trí 0.

### `/openarm/teleop/left_gripper` và `/openarm/teleop/right_gripper`

| Thuộc tính | Giá trị                                                                                          || ------------ | -------------------------------------------------------------------------------------------------- || Kiểu        | `std_msgs/msg/Float64MultiArray`                                                                 || `data[0]`  | hành trình ngón kẹp (m):`0.0` = đóng, `0.043` = mở hết. Các phần tử sau bị bỏ qua |

Giá trị ngoài khoảng 0.0 .. 0.043 bị clamp.

### Alias `JointState` (tuỳ chọn)

`/teleop/joint_commands` và `/meta/joint_states` nhận `sensor_msgs/msg/JointState`và **chỉ áp dụng cho khớp tay**:

- Có `name` (cùng số phần tử với `position`): áp dụng cho các khớp được nêu tên,  số lượng tuỳ ý. Chấp nhận tên như `left_j1`, `openarm_left_joint1`, `l_j1`... (xem  `JOINT_NAME_TO_ID` trong `sim/config.py`). Tên gripper bị bỏ qua.- Không có `name`: phải đủ 14 giá trị theo thứ tự như `/openarm/teleop/joint_commands`.

### Cách bridge gộp lệnh

1. Bridge giữ bảng mục tiêu mới nhất của từng khớp từ các topic teleop ở trên.2. Mỗi khi một topic có message mới, bridge gửi toàn bộ bảng (theo thứ tự 16 khớp,   chỉ gồm các khớp đã từng nhận lệnh) xuống backend qua UDP `127.0.0.1:9870`.   Khớp chưa từng nhận lệnh teleop thì giữ nguyên mục tiêu hiện tại. Ví dụ: chưa gửi   gripper thì gripper không bị kéo về 0.3. Backend clamp theo `JOINT_LIMITS`, cập nhật mục tiêu, rồi publish kết quả lên   `/openarm/joint_commands`.

Lệnh chỉ được gửi xuống motor thật khi server chạy `--real` và chế độ thực thi là`real` hoặc `dual`. Ở chế độ `sim` chỉ mô hình 3D chuyển động. Ngoài ra, backend bỏqua lệnh cho một motor cho tới khi đã đọc được vị trí thật của motor đó (log`[Safety] ... not synchronized yet`).

## 4. Lệnh thử nghiệm

Chạy sau khi đã `source /opt/ros/humble/setup.bash`. Khi server chạy `--real` ở chếđộ `real`/`dual`, các lệnh này **sẽ làm robot thật chuyển động**.

```bash# Xem trạng tháiros2 topic echo --once /openarm/joint_statesros2 topic hz /openarm/joint_states

# Tay trái tới một tư thế, tay phải về 0, publish liên tục 50 Hzros2 topic pub -r 50 --qos-reliability best_effort /openarm/teleop/joint_commands \  std_msgs/msg/Float64MultiArray \  "{data: [0.4, -0.4, 0.8, 0.5, 0.6, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}"

# Mở hết / mở 20 mm / đóng gripper tráiros2 topic pub --once --qos-reliability best_effort /openarm/teleop/left_gripper \  std_msgs/msg/Float64MultiArray "{data: [0.043]}"ros2 topic pub --once --qos-reliability best_effort /openarm/teleop/left_gripper \  std_msgs/msg/Float64MultiArray "{data: [0.020]}"ros2 topic pub --once --qos-reliability best_effort /openarm/teleop/left_gripper \  std_msgs/msg/Float64MultiArray "{data: [0.0]}"

# Alias JointState: chỉ di chuyển vài khớp theo tênros2 topic pub --once /teleop/joint_commands sensor_msgs/msg/JointState \  "{name: ['right_j1', 'right_j4'], position: [0.3, 0.9]}"```

Script có sẵn:

- `tools/test_meta_trajectory.py`: stream quỹ đạo giả lập lên `/teleop/joint_commands`.- `tools/demo_ros2_sequence.py`: chuỗi 7 tư thế, gồm cả gripper.