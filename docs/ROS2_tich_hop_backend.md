# openarm_shadow_ros: teleop OpenArm bằng camera, thay teleop Meta Quest

Tài liệu này gửi nhóm backend OpenArm để đối chiếu giao diện ROS 2 và xác nhận các giả định trước khi chạy robot thật.
Phần **7. Cần bên backend xác nhận** là phần quan trọng nhất.

- Mã nguồn: repo `24062005dang/openarm_shadow`, nhánh `Duc_ros_2`
  - package ROS 2: `ros2/openarm_shadow_ros/` (ament_python, node `teleop`, `launch/teleop.launch.py`)
  - phần nói chuyện với backend: `openarm_shadow/robot/ros2_bridge.py`
  - hướng dẫn cài và chạy: `docs/ROS2.md`
- Đối chiếu với backend: nhánh **`hoang1`**, các file `sim/openarm_joint_bridge.py`, `sim/config.py`,
  `sim/dashboard_server.py`.

## 1. Hệ thống làm gì

```
 3 camera (webcam + D455 + D435i)
   -> MediaPipe Pose + Hand, hợp nhất 3D nhiều camera
   -> retarget sang 7 góc khớp mỗi tay (theo URDF chính thức OpenArm v1.0) + độ mở kẹp
   -> lọc nhiễu, giới hạn góc / tốc độ, ly hợp (engage), dead-man
   -> ROS 2: /openarm/teleop/joint_commands, left_gripper, right_gripper  ──>  backend OpenArm (của các bạn)
 <- ROS 2: /openarm/joint_states  (đồng bộ tư thế, kiểm tra an toàn)
```

Bên mình **không** đụng CAN, không bật/tắt motor, không đổi gain hay giới hạn tốc độ của backend. Node chỉ đọc
`joint_states` và phát 3 topic lệnh.

## 2. Giao diện ROS 2

| Hướng | Topic | Kiểu | QoS bên mình | Tần số |
| --- | --- | --- | --- | --- |
| nhận | `/openarm/joint_states` | `sensor_msgs/JointState` | `BEST_EFFORT`, depth 10 | theo backend (100 Hz) |
| phát | `/openarm/teleop/joint_commands` | `std_msgs/Float64MultiArray` | `BEST_EFFORT`, depth 10 | 100 Hz khi đang engage |
| phát | `/openarm/teleop/left_gripper` | `std_msgs/Float64MultiArray` | `BEST_EFFORT`, depth 10 | khi đổi mức, hoặc mỗi 0,5 s |
| phát | `/openarm/teleop/right_gripper` | như trên | như trên | như trên |

- `joint_commands`: `data` = **đúng 14 số (rad)**, thứ tự `left_j1..left_j7` rồi `right_j1..right_j7`; `layout` để
  trống.
- Kẹp: `data[0]` = hành trình ngón (m), 0,0 = đóng, 0,043 = mở hết, **chia 10 mức đều**: 0; 4,78; 9,56; 14,33;
  19,11; 23,89; 28,67; 33,44; 38,22; 43 mm. Chỉ gửi đúng các giá trị này. Đổi mức có vùng trễ (phải lệch quá 0,8
  bước) để kẹp không nhảy qua lại khi tay ở gần ranh giới. Số mức chỉnh trong config (`ros2.gripper.levels`).
- `joint_states`: đọc theo **tên** (`left_j1..left_j7`, `left_gripper`, `right_j1..right_j7`, `right_gripper`),
  không phụ thuộc thứ tự. Không dùng `velocity`, `effort`, `header.stamp`. Độ "tươi" tính theo lúc nhận được.
- Không dùng `/openarm/joint_commands`, `/teleop/joint_commands`, `/meta/joint_states`, `/openarm/joint_trajectory`.
- Tên node: `openarm_shadow_teleop`.

## 3. Quy ước góc (giả định, cần xác nhận ở mục 7)

- Theo tài liệu và code nhánh `hoang1`: góc khớp trên mọi topic = **góc motor thô theo URDF chính thức**, không offset,
  không đổi dấu, không quy về ±180°. Giới hạn theo `JOINT_LIMITS` trong `sim/config.py` (vd `left_j1` −3,4907..1,3963,
  `right_j1` −1,3963..3,4907).
- Bên mình tính góc theo đúng URDF đó (dữ liệu động học trích từ `enactic/openarm_description`, `v1.urdf`), nên
  **gửi thẳng, không quy đổi**. Có sẵn bảng bù `sign` / `offset` từng khớp trong config, mặc định không dùng, chỉ để
  chỉnh nhanh khi đo thấy lệch mà không phải sửa code.
- Ở tư thế tay thả xuôi, bên mình coi mọi khớp = 0.

## 4. Hành vi theo thời gian

| Lúc | Node làm gì |
| --- | --- |
| Khởi động | Chờ `/openarm/joint_states` đủ khớp (tối đa 5 s), không thì báo lỗi và thoát |
| Kiểm tra an toàn | Nếu góc đo nằm ngoài giới hạn mềm của bên mình quá 5° (vd `left_j1` ≈ 3,1 rad lúc tay thả xuôi), **không bao giờ phát lệnh** |
| Chưa engage (người điều khiển chưa vào tư thế, đang hiệu chuẩn) | **Không phát gì.** Trạng thái bên trong bám theo `joint_states`: robot được đẩy tay hay được web UI điều khiển thì bên mình đi theo, không kéo lại |
| Engage (người giữ tư thế READY 2 s, hoặc bấm SPACE) | Phát 14 khớp 100 Hz, bắt đầu **đúng tư thế đang đo** (không giật), tăng tốc mềm vài giây |
| Chỉ điều khiển 1 tay | Tay còn lại gửi đúng góc đo hiện tại của nó (để backend giữ yên, không kéo về 0) |
| Mất người/tay trong ảnh | Khớp tương ứng giữ giá trị cũ; quá 0,4 s không có dữ liệu mới thì đứng yên hẳn |
| Nhả (SPACE) | Ngừng phát. Backend giữ mục tiêu cuối |
| `joint_states` ngừng > 0,5 s khi đang engage | Ngừng phát và thoát. Backend giữ mục tiêu cuối |
| Thoát | Ngừng phát, không gửi tư thế nghỉ (backend giữ tư thế). Muốn về nghỉ thì người vận hành bấm `p` trước khi thoát |
| `--dry-run` | Chỉ đọc và vẽ, không bao giờ phát |

Giới hạn phía bên mình (lớp thêm vào, không thay giới hạn của backend):
- Góc mềm (độ, URDF): phải `[[-75,75],[-9,90],[-85,85],[0,135],[-85,85],[-40,40],[-80,80]]`; trái giống hệt, chỉ
  khác J2 là `[-90,9]`.
- Tốc độ lúc chạy thử: 20°/s cho J1–J4, 15°/s cho J5–J7.
- Chống hai tay va nhau (capsule).

## 5. Chạy trên máy bên backend

Ubuntu 22.04, ROS 2 Humble, Python 3.10. Cần cắm 3 camera vào máy chạy và hiệu chuẩn lại camera tại chỗ.

```bash
source /opt/ros/humble/setup.bash && source ~/shadow_ws/install/setup.bash
export OPENARM_SHADOW_ROOT=~/openarm_shadow
ros2 run openarm_shadow_ros teleop --source multi --dry-run --config ~/openarm_shadow/config/local_3cam.yaml
# hoặc
ros2 launch openarm_shadow_ros teleop.launch.py configs:="<các file config>" dry_run:=true
```

Chi tiết cài đặt (apt, pip, colcon) và quy trình chạy thật từng bước ở `docs/ROS2.md`.

## 6. Đã kiểm thử (chưa có robot thật)

Môi trường: container `ros:humble-ros-base` (Python 3.10.12). App chạy thật, nhưng thay camera bằng dữ liệu thô 3
camera đã ghi. Nối với một backend giả viết theo `openarm_joint_bridge.py` + bộ nội suy nhánh `hoang1`: 0,25 rad/s,
giữ mục tiêu cuối, kẹp theo `JOINT_LIMITS`.

- Backend nhận 1137/1137 lệnh, 97,4 Hz, khoảng cách lớn nhất 18,5 ms. Mọi lệnh đúng 14 số, không NaN, trong
  `JOINT_LIMITS`.
- Lệnh kẹp: ~54 lệnh mỗi tay trong ~13 s.
- Không có lệnh nào trước khi engage. Lệnh đầu tiên lệch tư thế đo 0,05°.
- Khớp bị đẩy tới 0,6 rad trước khi engage thì lệnh đầu tiên là 0,597 rad.
- `--dry-run`: 0 lệnh. Backend dừng giữa chừng: node ngừng gửi sau 0,5 s.

Kiểm tra luồng riêng (05/10, container Humble): `ros2 topic list -t` thấy đúng 4 topic và kiểu ở mục 2.
`ros2 topic echo` lệnh khớp ra đúng 14 số. Backend giả nhận 393/393 lệnh, 98,1 Hz, khoảng cách lớn nhất 11,9 ms. Kẹp
mở dần 0 → 1 phát đúng 10 lệnh mỗi tay, đúng 10 mức 0; 4,78; ...; 43 mm.

Các bạn tự chạy lại được vòng test này (cần Docker): `tools/ros2_loop_test.sh`. Backend giả ở
`scripts/fake_openarm_backend.py`, phân tích ở `scripts/analyze_ros2_log.py`. Thay backend giả bằng backend thật
ở chế độ `sim` cũng được.

## 7. Cần bên backend xác nhận

1. **Nhánh chạy trên máy robot là `hoang1`**, và tên topic / kiểu / QoS ở mục 2 khớp với bản đang chạy?
2. **Robot thả xuôi hai tay thì `/openarm/joint_states` đọc mọi khớp ≈ 0?** Nhờ gửi kết quả
   `ros2 topic echo --once /openarm/joint_states` ở tư thế này.
   - Robot bên mình đo trực tiếp qua CAN thấy motor J1, J2 tay trái đọc khoảng **178°** ở tư thế này, do motor lắp lệch
     180°. Bản chính thức `openarm_driver` có offset π cho hai khớp này.
   - Nếu `left_j1` / `left_j2` của các bạn ra khoảng ±3,1 rad, thì lệnh theo URDF sẽ bị `JOINT_LIMITS` kẹp và tay trái
     quay rất lớn. Bên mình chặn sẵn trường hợp này (không phát lệnh), nhưng cần thống nhất cách bù.
3. **Kẹp:** cả hai tay đều 0 m = 0 rad (đóng), 0,043 m = −1,20 rad (mở), và zero kẹp hai bên đã hiệu chuẩn như vậy?
   Robot bên mình đo được zero kẹp trái khác kẹp phải khoảng 50°.
4. **Một nguồn lệnh duy nhất:** khi node này chạy thì không chạy teleop Meta Quest / script khác cùng phát vào
   `/openarm/teleop/*`, đúng không? Bridge gộp mọi nguồn vào cùng một bảng mục tiêu.
5. **Tốc độ:** lệnh từ topic teleop có đi qua `velocity_limit` (mặc định 0,25 rad/s) như bọn mình hiểu? Bọn mình đề
   nghị giữ 0,25 rad/s cho lần chạy đầu, sau đó tăng dần (vd 0,5 → 1 rad/s) để robot theo kịp người.
6. **Tần số:** 100 Hz cho `joint_commands` (mỗi bản tin là một gói UDP sang backend) và kẹp chỉ gửi khi đổi mức
   (thêm 2 lệnh/giây giữ kết nối) có ổn không? Kẹp 10 mức như trên đúng yêu cầu chưa?
7. **Bật motor / E-stop:** bên backend bật motor và giữ E-stop; node bên mình chỉ phát khi người vận hành engage.
   Cách chia việc này có khớp quy trình của các bạn không?
8. **Chạy chung máy:** node dùng nhiều CPU (MediaPipe trên 3 camera). Bộ nội suy 400 Hz của backend có bị ảnh hưởng
   không? Có cần đặt `ROS_DOMAIN_ID` riêng không?
9. **Dataset ACT:** nếu các bạn ghi `/openarm/joint_commands` làm `action`, thì lệnh của bọn mình đã qua lọc và giới
   hạn tốc độ (mượt, có trễ khoảng 0,2–0,5 s so với người). Như vậy có phù hợp không?

Nếu có điểm nào khác với hiểu biết ở trên, bọn mình chỉ cần sửa config (`ros2:` trong `config/default.yaml`), không
phải sửa package.
