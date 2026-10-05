# Chạy openarm_shadow như một package ROS 2 (thay teleop Meta Quest)

Hệ thống của ta chỉ làm **camera → nhận diện → retarget → lệnh khớp**. Điều khiển motor, CAN, giới hạn tốc độ, giao
diện web là việc của backend OpenArm bên kia (repo `hoanglmv/openarm_can`, **nhánh `hoang1`**:
`sim/openarm_joint_bridge.py` + `sim/dashboard_server.py`).

## 1. Giao diện

| Hướng | Topic | Kiểu | Nội dung |
| --- | --- | --- | --- |
| nhận | `/openarm/joint_states` | `sensor_msgs/JointState` | 16 tên `left_j1..left_j7, left_gripper, right_j1..right_gripper`; khớp rad, kẹp m |
| phát | `/openarm/teleop/joint_commands` | `std_msgs/Float64MultiArray` | **đúng 14 số** (rad): `left_j1..j7`, rồi `right_j1..j7` |
| phát | `/openarm/teleop/left_gripper` | `std_msgs/Float64MultiArray` | `data[0]` = hành trình kẹp (m): 0 = đóng, 0,043 = mở hết |
| phát | `/openarm/teleop/right_gripper` | như trên | |

- QoS: phát `BEST_EFFORT` depth 10 (khớp subscriber của bridge), nhận `BEST_EFFORT` (nhận được từ publisher `RELIABLE`).
- Tần số: lệnh khớp phát mỗi nhịp vòng điều khiển, **100 Hz** (đo trong thử nghiệm: 97 Hz).
- **Quy ước góc:** theo backend: URDF chính thức của OpenArm, góc khớp = góc motor thô (không offset, không đổi dấu,
  không quy về ±180°). Pipeline của ta tính theo đúng URDF đó nên gửi thẳng. `ros2.urdf_to_ros` (sign, offset) trong
  `config/default.yaml` mặc định đồng nhất, chỉ để chỉnh khi đo thấy lệch.
- Kẹp: độ mở 0..1 của ta làm tròn về `ros2.gripper.levels` mức (mặc định **10**: 0; 4,78; ...; 43 mm) ngay trước khi
  phát, có vùng trễ `level_hysteresis` (0,3 bước). Backend tự đổi sang góc motor (0 → −1,20 rad).

## 2. Hành vi

| Tình huống | Node làm gì |
| --- | --- |
| Khởi động | Chờ `/openarm/joint_states` (tối đa 5 s). Góc đo ngoài `safety.soft_limits_deg` quá 5° → **không cho gửi lệnh** (quy ước góc / zero robot khác URDF) |
| Chưa engage (đang hiệu chuẩn tay, chờ READY, đã nhả SPACE) | **Không phát gì.** Lệnh bên trong bám theo góc đo: robot bị đẩy tay hay được backend điều khiển thì hình trong app đi theo; lúc engage lệnh bắt đầu đúng tư thế thật |
| Engage (giữ READY `auto_engage_real_s` giây hoặc bấm SPACE) | Phát 14 khớp 100 Hz qua SafetyGate (giới hạn góc/tốc độ của ta, tăng tốc mềm, dead-man) |
| Chỉ điều khiển 1 tay (`--arms right`) | Tay kia gửi đúng góc đo hiện tại (backend không kéo tay đó đi đâu) |
| Mất người / mất tay trong ảnh | Khớp tương ứng giữ nguyên (SafetyGate); backend giữ mục tiêu cuối |
| Kẹp | Làm tròn về 10 mức; chỉ phát khi đổi mức hoặc mỗi 0,5 s (backend in một dòng log cho mỗi lệnh kẹp) |
| `joint_states` ngừng > 0,5 s khi đang engage | Ngừng gửi, thoát (`LỖI vòng điều khiển ...`); backend giữ mục tiêu cuối |
| `--dry-run` | Chỉ đọc `joint_states` và vẽ (nét xanh lá = robot thật); không bao giờ phát |
| Thoát (q / Esc) | Ngừng phát. Không đưa tay về nghỉ (backend giữ tư thế); muốn về nghỉ thì bấm `p` trước khi thoát |

Lưu ý tốc độ: backend tự giới hạn **0,25 rad/s (~14°/s)** mặc định, chặt hơn giới hạn của ta (20°/s). Robot sẽ chậm hơn
khi chạy với CAN trực tiếp. Chỉnh trên web UI của backend (`set_velocity_limit`, 0,02–3 rad/s) khi đã chạy ổn.

## 3. Cài trên máy chạy (Ubuntu 22.04, ROS 2 Humble, Python 3.10)

```bash
# 1. Mã nguồn
git clone -b Duc_ros_2 https://github.com/24062005dang/openarm_shadow.git ~/openarm_shadow
cd ~/openarm_shadow
bash scripts/download_models.sh                        # model MediaPipe vào models/

# 2. Thư viện (Python của ROS, không dùng venv riêng để thấy rclpy)
sudo apt install -y python3-pip libgl1 libglib2.0-0 libegl1 libgles2
pip3 install --user "numpy<2" mediapipe==0.10.21 pyrealsense2==2.58.1.10581 pyyaml

# 3. Package ROS 2
mkdir -p ~/shadow_ws/src && ln -s ~/openarm_shadow/ros2/openarm_shadow_ros ~/shadow_ws/src/
cd ~/shadow_ws && source /opt/ros/humble/setup.bash && colcon build --packages-select openarm_shadow_ros

# 4. Mỗi terminal
source /opt/ros/humble/setup.bash && source ~/shadow_ws/install/setup.bash
export OPENARM_SHADOW_ROOT=~/openarm_shadow             # node tìm thư viện openarm_shadow ở đây
```

Camera: cắm webcam + D455 + D435i vào máy chạy, rồi kiểm tra bằng `python3 scripts/list_cameras.py`. Sau đó sửa
serial và chỉ số webcam trong config camera (vd `config/local_3cam.yaml`) và **hiệu chuẩn lại camera** trên máy đó
(`scripts/calibrate_cameras.py`, xem `lenh_duc.md`). File hiệu chuẩn của máy cũ chỉ đúng nếu camera không xê dịch.

GPU: MediaPipe mặc định chạy CPU. Thử `models.delegate: gpu` sau khi đo bằng `scripts/bench_mediapipe.py`.

## 4. Quy trình chạy (theo thứ tự, có người giữ E-stop)

```bash
# 0. Backend OpenArm (bên kia) đang chạy chế độ real, robot đã sync 16 motor.
ros2 topic hz /openarm/joint_states                    # ~100 Hz

# 1. Robot thả xuôi hai tay: mọi khớp phải gần 0 (left_j1, left_j2 không được gần ±3,1)
ros2 topic echo --once /openarm/joint_states

# 2. Chỉ đọc: cầm tay robot di chuyển từng khớp, nét xanh lá phải đi đúng chiều
ros2 run openarm_shadow_ros teleop --source multi --dry-run \
    --config ~/openarm_shadow/config/local_3cam.yaml

# 3. Chạy thật một tay, chậm (giữ READY để engage, SPACE nhả, q thoát)
ros2 run openarm_shadow_ros teleop --source multi --arms right \
    --config ~/openarm_shadow/config/d455_wrist_real.yaml --config ~/openarm_shadow/config/gripper_real.yaml \
    --config ~/openarm_shadow/config/local_both_full.yaml --config ~/openarm_shadow/config/local_3cam.yaml \
    --record run_ros_right.npz

# 4. Hai tay (bỏ --arms), rồi mới tăng tốc độ trên web UI backend
```

Chạy bằng launch file (không hỏi `yes`, vẫn phải giữ READY hoặc bấm SPACE mới engage):

```bash
ros2 launch openarm_shadow_ros teleop.launch.py \
    configs:="$HOME/openarm_shadow/config/d455_wrist_real.yaml $HOME/openarm_shadow/config/local_3cam.yaml" \
    dry_run:=true
```

Không cần colcon cũng chạy được: `python3 scripts/shadow.py --robot ros2 ...` (sau khi `source /opt/ros/humble/setup.bash`).

Sau mỗi lần chạy thật: `python3 scripts/measure_lag.py run_ros_right.npz` và `scripts/find_jumps.py`. So với số đo
khi chạy CAN trực tiếp (`data/2026-10-03_hardware_baseline/`).

## 5. Vòng test 100 Hz không cần robot

Chạy node thật với dữ liệu thô đã ghi thay camera (`--source replay:<thư mục>`), nối với backend giả trong container
ROS 2 Humble, rồi phân tích mọi lệnh đã phát: tần số, bước nhảy, tốc độ / gia tốc từng khớp, giới hạn khớp của backend,
mức kẹp, độ lệch robot - lệnh. Có đồ thị.

```bash
tools/ros2_loop_test.sh data/2026-10-03_raw_bimanual_take2 /tmp/ros2_test \
    --replay-intrinsics /raw/intrinsics_approx.yaml \
    --config config/d455_wrist_real.yaml --config config/gripper_real.yaml \
    --config config/local_both_full.yaml --config config/local_3cam.yaml
# kết quả: /tmp/ros2_test/bao_cao.txt, ros2_lenh.png, backend.npz, shadow.npz
```

Có ROS 2 sẵn trên máy thì chạy trực tiếp: `scripts/fake_openarm_backend.py --out backend.npz --push` ở một terminal,
`scripts/shadow.py --robot ros2 --source replay:<thư mục> ...` ở terminal khác, rồi `scripts/analyze_ros2_log.py`.

## 6. Đã kiểm thử (05/10/2026, container `ros:humble-ros-base`, Python 3.10.12)

- 172 test đạt; `colcon build`, `ros2 run ... --help`, `ros2 launch ... --show-args` chạy được.
- App thật (dữ liệu thô 3 camera ghi ngày 03/10) + backend giả mô phỏng bridge nhánh `hoang1`:
  - backend nhận 1137/1137 lệnh, **97,4 Hz**, khoảng cách lớn nhất 18,5 ms; mọi lệnh đúng 14 số, không NaN, trong
    `JOINT_LIMITS` của backend; kẹp ~54 lệnh mỗi tay trong ~13 s;
  - không phát gì trước khi engage; lệnh đầu tiên lệch tư thế thật 0,05°; khuỷu trái bị "đẩy" tới 0,6 rad trước khi
    engage thì lệnh đầu tiên là 0,597 rad (không kéo tay về tư thế cũ);
  - `--dry-run`: 0 lệnh; backend dừng giữa chừng: app ngừng gửi sau 0,5 s và thoát.
- **Chưa** thử với robot thật và backend thật.
