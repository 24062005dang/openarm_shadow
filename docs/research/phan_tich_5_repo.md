# Teleop OpenArm – Phân tích 5 repo & dịch 2 bài báo

Sep 28, 2026 · @Dang

## Cách đọc tài liệu này

Tài liệu gồm hai phần. Phần đầu phân tích 5 repo bạn chọn, đọc thẳng từ mã nguồn trên GitHub. Phần sau dịch sang tiếng Việt 2 bài báo SEW-Mimic và Vision-Based Hand Shadowing, dịch từng mục theo đúng thứ tự trong bài.

Quy ước đánh dấu:

- **★ OpenArm v1.0:** chỗ có liên quan trực tiếp tới robot của nhóm, nên giữ lại khi bạn lọc.
- **(Ghi chú: …)** trong phần dịch: lời giải thích của mình, không có trong bài gốc.
- Thuật ngữ tiếng Anh được giữ trong ngoặc ở lần xuất hiện đầu, ví dụ "động học ngược (inverse kinematics, IK)".

## So sánh nhanh 5 repo

Không repo nào làm sẵn đúng việc "camera → 7 góc khớp → OpenArm". Repo 3 (Marionette) có phần xử lý tín hiệu tốt nhất; repo 1 là repo duy nhất chạy trên OpenArm v10 thật; repo 5 là cách duy nhất ra đủ 7 góc tay nhưng chạy offline.

| Repo | Mô hình pose | Cách ra lệnh robot | Số khớp điều khiển | Robot đích | Thời gian thực | License | Mức dùng lại cho OpenArm v1.0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1. [openarm\_teleoperation](https://github.com/ahsanali555/openarm_teleoperation) | MediaPipe Pose (2 điểm cổ tay) | Vận tốc cổ tay 2D → MoveIt Servo giải IK | 7 (do Servo chọn) | OpenArm v10 thật | Có, 30 Hz | Không có | Cao về bring-up ROS 2; thấp về cách ánh xạ |
| 2. [robot-teleop](https://github.com/jasongedev/robot-teleop) | Lightweight 3D pose (GPU) | Tính góc vai, khuỷu → gửi góc khớp | 4 mỗi tay + thân | Motoman SDA10F (mô phỏng) | Có | Không có | Trung bình: đúng ý tưởng, code có lỗi, ROS 1 cũ |
| 3. [Teleoperation-Arm](https://github.com/Im-ma/Teleoperation-Arm) | MediaPipe Pose + Hand (trình duyệt) | Góc trong ảnh + IK 2 khâu → góc khớp | 5 + kẹp | SO-101 thật | Có, 30 Hz | Không có | Cao về lọc, tin cậy, an toàn; phải viết lại phần ánh xạ |
| 4. [kinetic-arm-lab](https://github.com/YashVG/kinetic-arm-lab) | MediaPipe Pose Lite (trình duyệt) | Vị trí cổ tay → IK 3 khớp | 3 | Tay ảo | Có | Không có | Trung bình: ly hợp, xử lý mất tracking, test |
| 5. [pose2sim](https://github.com/perfanalytics/pose2sim) | RTMPose, nhiều camera | Tam giác hoá 3D → IK OpenSim ra góc khớp | 7 mỗi tay | Không có robot | Không (offline) | BSD-3 | Cao cho ground truth và thu động tác Thái Cực |

## Repo 1 — ahsanali555/openarm\_teleoperation

Repo này chạy trên đúng OpenArm v10 (v1.0) thật, nhưng chỉ dùng cổ tay trong ảnh như một cần điều khiển tốc độ 2D. Nó **không tính góc khớp tay người**. [Link repo](https://github.com/ahsanali555/openarm_teleoperation), commit 668bb6d ngày 30/06/2026.

| Mục | Nội dung |
| --- | --- |
| Nền tảng | ROS 2 Humble, Ubuntu 22.04, MoveIt 2 Servo, ros2\_control, stack `enactic/openarm_ros2` |
| Camera, mô hình | 1 webcam, MediaPipe Pose (`model_complexity=1`), chỉ dùng điểm 15, 16 (hai cổ tay) |
| Đầu ra | Vận tốc đầu công tác (Twist) theo trục Y (ngang) và Z (dọc); X và xoay bằng 0 |
| Robot | OpenArm v10 hai tay, CAN FD 1 Mbps / 5 Mbps |
| Nhánh thứ hai | Cử chỉ tay nhận bằng STM32 + cảm biến ToF, gọi 7 tư thế có sẵn |
| License | Không có file license, README ghi "All Rights Reserved" |

### Cách hoạt động (bản v4, bản đang chạy)

File `pose_teleop_node.py` dài 2064 dòng nhưng chứa 5 phiên bản. Bản 1, 2, 3, 5 bị bọc trong chuỗi `'''` nên không chạy; chỉ bản v4 (dòng 946–1478) chạy.

1. Đọc webcam, lật gương ảnh, chạy MediaPipe Pose. Chỉ nhận cổ tay có độ tin cậy (visibility) từ 0.55.
2. Khi thấy đủ hai cổ tay trong 15 khung liên tiếp, chốt vị trí lúc đó làm "điểm gốc".
3. Độ lệch cổ tay so với điểm gốc (toạ độ ảnh chuẩn hoá 0–1) được kẹp trong ±0.20, bỏ vùng chết 0.03, nhân hệ số 0.8.
4. Độ lệch ngang → vận tốc trục Y của robot, độ lệch dọc → vận tốc trục Z. Đây là **điều khiển tốc độ** giống cần joystick: tay lệch càng xa thì robot chạy càng nhanh, không phải bắt chước vị trí.
5. Lọc EMA (hệ số 0.6), gửi 30 Hz tới MoveIt Servo. Servo tự giải IK bằng Jacobian và xuất `JointTrajectory` 20 Hz cho ros2\_control.
6. Nếu cổ tay ra khỏi hộp ±0.20 hoặc mất tracking: bỏ điểm gốc, đưa hai tay về tư thế nhà `[0, ±0.35, 0, 2.0, 0, 0, 0]` rad trong 3 s, rồi bắt đầu lại.
7. Lúc khởi động, file launch đưa tay về tư thế hạt giống không kỳ dị, 4 s sau mới bật Servo, 13 s sau mới bật teleop.

Bản v5 (đang bị comment) đổi hướng: gửi vận tốc thẳng cho từng khớp (`JointJog`), chỉ J1, J2, J4. Ảnh ngang → J1, ảnh dọc → J2, còn J4 mới là khung rỗng. Tác giả ghi lý do: không dùng IK thì không lo kỳ dị.

### Đánh giá cho nhóm

- **★ OpenArm v1.0 — nên lấy:** lệnh bring-up v10 qua `openarm_ros2` (`arm_type:=v10`), giá trị tư thế nhà không kỳ dị, cơ chế tự về nhà khi mất tracking, HUD vẽ hộp làm việc, cách ghi log độ trễ ra CSV.
- **Không khớp ý tưởng của nhóm:** chỉ dùng 2 điểm cổ tay, bỏ vai, khuỷu, bàn tay; không có độ sâu; không điều khiển hướng cổ tay. Dáng tay robot do IK của Servo tự chọn, không giống dáng tay người.
- **Rủi ro:** `check_collisions: false`, ngưỡng kỳ dị nâng lên 50/100 (mặc định 17/30). `trajectory_recorder.py` dùng FK phẳng 3 khâu tạm, không đúng OpenArm.
- **Môi trường:** repo viết cho ROS 2 Humble (Ubuntu 22.04). Máy nhóm là Ubuntu 24.04 nên phải dùng ROS 2 Jazzy; mình chưa kiểm tra `openarm_ros2` có hỗ trợ Jazzy không.
- **Bản quyền:** không có license, nên đọc để học ý tưởng, không chép nguyên code vào repo của nhóm.

## Repo 2 — jasongedev/robot-teleop

Repo này làm đúng ý tưởng của nhóm: ước lượng xương 3D từ một webcam, tính góc vai và khuỷu, gửi thẳng cho hai tay robot 7 khớp. Nhưng code cũ (2020) và cách tính góc có lỗi, nên chỉ nên học ý tưởng. [Link repo](https://github.com/jasongedev/robot-teleop), commit 46da8c3 ngày 05/11/2020.

| Mục | Nội dung |
| --- | --- |
| Nền tảng | Ubuntu 18.04, ROS 1 Melodic, ROS-Industrial, MoveIt, PyTorch ≥ 1.6, cần GPU CUDA |
| Mô hình pose | Lightweight 3D human pose (MobileNet, 19 điểm kiểu CMU Panoptic), file `human-pose-estimation-3d.pth` có sẵn trong repo |
| Đầu ra | 9 góc: mỗi tay 2 góc vai + 2 góc khuỷu, cộng 1 góc xoay thân |
| Robot | Yaskawa Motoman SDA10F (2 tay × 7 khớp + thân), chạy trong mô phỏng |
| License | Không có file license |

### Cách hoạt động

Luồng xử lý theo sơ đồ trong README: camera → mô hình pose 3D → `joint_angle_calculator` → gửi qua TCP → node ROS `custom_joint_mover` → `joint_trajectory_action` → MoveIt → driver robot.

1. `inference.py` chạy mạng pose 3D trên từng khung ảnh, ra 19 điểm 3D (đơn vị cm).
2. Dùng ma trận ngoại tham `extrinsics.json` (R, t) để xoay bộ xương từ khung camera sang khung thế giới, rồi đổi trục (x, y, z) → (−z, x, −y).
3. `joint_angle_calculator.py` tính góc:
   - **Vai:** lấy vector vai→khuỷu, tính `roll = atan2(|dx|, |dy|) − π/2` và `pitch = atan2(|dz|, |dy|) − π/2`.
   - **Khuỷu:** góc giữa cánh tay và cẳng tay bằng tích vô hướng (`arccos`), cộng một góc xoay `psi` tính từ trục tích có hướng.
   - **Thân:** so trung bình toạ độ các điểm bên trái và bên phải để ước lượng góc xoay người.
4. 9 góc được gửi dạng chuỗi qua TCP cổng 8082. `custom_joint_mover.py` nhận, gán vào khớp S, L, E, U của mỗi tay và khớp thân B1. Ba khớp cổ tay R, B, T luôn bằng 0.
5. Góc được gửi thành một điểm quỹ đạo duy nhất với `time_from_start = 0`. README gợi ý nâng tốc độ cập nhật lên 100 Hz trong mô phỏng.

### Lỗi và điểm yếu mình thấy khi đọc code

- **Mất dấu góc vai:** `abs()` trên dx, dy, dz khiến đưa tay ra trước và ra sau cho cùng một góc.
- **Chỉ có 4 khớp mỗi tay:** không có xoay cánh tay đúng nghĩa và không có cổ tay.
- **Không lọc nhiễu ở đầu ra:** repo có file One Euro filter nhưng góc gửi đi không qua lọc, không kẹp giới hạn khớp, không giới hạn tốc độ.
- **Không an toàn:** `eval()` chạy thẳng chuỗi nhận qua mạng; mỗi khung mở một kết nối TCP mới; `sock.close` thiếu dấu `()` nên socket không được đóng.
- **Tiêu cự cố định `fx = 1`:** nhánh tự ước lượng tiêu cự không bao giờ chạy, nên độ sâu gốc của xương có thể sai.

### Đánh giá cho nhóm

- **★ OpenArm v1.0 — nên lấy:** ý tưởng kiến trúc (pose → tính góc → gửi góc khớp), bước hiệu chuẩn ngoại tham camera bằng (R, t), cách tính góc khuỷu bằng tích vô hướng.
- **Phải viết lại:** góc vai có dấu theo đúng thứ tự trục OpenArm (nâng trước → dang → xoay), thêm J3 và J5–J7, thêm lọc, kẹp giới hạn, giới hạn tốc độ.
- **Môi trường:** ROS 1 Melodic không cài được trên Ubuntu 24.04; mô hình cần GPU CUDA mà laptop nhóm không có. Thay bằng MediaPipe Pose (có điểm 3D thế giới, chạy CPU).

## Repo 3 — Im-ma/Teleoperation-Arm (Marionette)

Repo này có phần xử lý tín hiệu từ camera kỹ nhất trong 5 repo: đo góc, lọc nhiễu, giữ từng khớp khi mất tin cậy, và các lớp an toàn. Điểm hạn chế là nó viết cho tay SO-101 5 khớp. [Link repo](https://github.com/Im-ma/Teleoperation-Arm); nhánh `main` commit bb16171 ngày 26/09/2026 (đã gộp PR #6); bản cũ hơn nằm ở [PR #1](https://github.com/Im-ma/Teleoperation-Arm/pull/1).

| Mục | Nội dung |
| --- | --- |
| Bối cảnh | Làm trong hackathon HackGT 13 (26/09/2026), 3 người, kế hoạch 6 giờ |
| Nền tảng | Trình duyệt Chrome chạy MediaPipe; server Python `serve.py` (aiohttp, WebSocket); bản rút gọn của LeRobot cho motor Feetech |
| Mô hình | MediaPipe Tasks `@mediapipe/tasks-vision@0.10.21`: Pose Landmarker heavy (lỗi thì dùng full) + Hand Landmarker, chạy trên GPU của trình duyệt |
| Robot | SO-101: pan, lift, elbow, wrist flex, wrist roll, gripper |
| Tần số | Vòng điều khiển 30 Hz |
| License | Không có license ở gốc repo; riêng file URDF SO-101 trong `web/robot/` là Apache-2.0 |

Phần Gemini, ElevenLabs, Tiger Data, Vultr trong repo là tính năng dự giải tài trợ của hackathon, không liên quan điều khiển.

### Quá trình tiến hoá của cách ánh xạ

1. **Range sync (PR #1):** người dùng giữ 9 tư thế để ghi min/max mỗi đặc trưng, rồi kéo dãn vào 85% tầm hoạt động của robot. Nhóm tác giả ghi nhận: cách này không đảm bảo "tay thẳng lên thì robot thẳng lên".
2. **Góc-với-góc (PR #1, đề xuất):** chọn một tư thế chung làm gốc, sau đó 1° của người = 1° của robot, kẹp theo giới hạn.
3. **Marionette Live (main):** không cần hiệu chuẩn. Người dùng giữ tư thế "goalpost" (cánh tay ngang vai, cẳng tay dựng đứng) trong 0.5–1 s để khoá. Vai và khuỷu robot được tính bằng IK theo vị trí bàn tay; cổ tay theo hướng bàn tay; đế xoay theo hướng cánh tay.

### Cách đo góc (`web/angles.mjs`, `web/mapping.mjs`)

- **Khung thân trong ảnh:** trục x dọc đường nối hai vai, trục y hướng về phía mũi. Góc nâng tay và góc khuỷu đo trong mặt phẳng ảnh, nên không đổi khi người đứng xa hay gần. Điều này chỉ đúng khi trục bản lề của khớp hướng vào camera.
- **Trộn sang 3D khi tay chĩa về camera:** tính tỉ lệ `depth` = phần của đoạn vai→cổ tay nằm theo trục camera. Từ 0.3 tới 0.7 thì trộn dần từ góc 2D sang góc tính từ điểm 3D thế giới của MediaPipe.
- **Xoay đế (pan):** lấy từ điểm 3D; không xác định khi cánh tay gần thẳng đứng (thành phần ngang dưới 0.3), lúc đó giữ giá trị cũ. Bỏ qua khi người xoay thân quá 25°.
- **Cổ tay, xoay cổ tay, kẹp:** từ Hand Landmarker. Xoay cổ tay dùng đường nối gốc ngón trỏ–ngón út; kẹp dùng khoảng cách đầu ngón cái–ngón trỏ chia chiều dài bàn tay. Khi lòng bàn tay nghiêng quá khoảng 70° thì giữ góc xoay cũ, vì độ sâu của MediaPipe lúc đó hay lật dấu.
- **Sửa lỗi MediaPipe đổi nhãn trái/phải:** so vị trí hai vai trong ảnh để biết nhãn có bị đảo không; chọn bàn tay có cổ tay gần cổ tay của cánh tay đang theo dõi, không tin nhãn handedness.
- **Tay chuyển động nhanh:** cổ tay chạy nhanh hơn 1.5 lần bề rộng vai mỗi giây thì ảnh bị nhoè, giữ nguyên các khớp bàn tay.

### Lọc nhiễu (`web/filters.mjs`)

- Mỗi khớp một bộ lọc One Euro (Casiez, CHI 2012): êm khi đứng yên, nhanh khi di chuyển. Ví dụ lift, elbow dùng tần số cắt tối thiểu 0.4 Hz, beta 0.01.
- Vùng chết có trễ (hysteresis) theo từng khớp: 2° cho vai/khuỷu, 3–4° cho cổ tay, 8° cho pan.
- Nhảy quá 30° trong một khung coi là lỗi tracking: giữ 0.2 s (0.5 s với xoay cổ tay) rồi mới chấp nhận.
- Độ tin cậy từng khớp: dưới 0.6 thì riêng khớp đó đứng yên, các khớp khác vẫn chạy. Kẹp dùng trung vị 5 khung.

### An toàn và vòng điều khiển (`serve.py`)

- Vòng 30 Hz. Giới hạn bước 2.5°/tick khi bắt chước (khoảng 75°/s), 1.5°/tick khi tự về nhà, 3°/tick khi diễn cử chỉ. Mọi khớp được co cùng một tỉ lệ để bắt đầu và kết thúc cùng lúc.
- Dead-man 0.4 s: trang web không gửi mục tiêu mới thì robot đứng yên.
- Khi bắt đầu điều khiển: trộn từ tư thế hiện tại sang tư thế người theo đường cong smoothstep trong 1–3 s.
- Mỗi lần kết nối hoặc tải trang: robot về tư thế sẵn sàng cố định. Không gì từ máy chủ cloud được phép di chuyển robot.
- Máy trạng thái (`web/mirror.mjs`): BOOT → HOMING → WAITING → ACQUIRING → MIRRORING ↔ HOLD. Mất tracking quá 400 ms thì HOLD; muốn quay lại phải đưa tay về trong vòng 5 cm quanh tay robot và giữ 0.5 s.

### Bài học tác giả ghi lại (`docs/RESEARCH.md` trong PR #1)

- `mediapipe` Python 1.0.x đã bỏ `mp.solutions`; bản 0.10.21 chạy được.
- Trình duyệt chỉ cho mở camera trên `localhost` hoặc HTTPS; dùng điện thoại qua mạng LAN phải có tunnel hoặc chứng chỉ.
- Xoay cổ tay nằm ngay chỗ encoder ±180° khiến robot xoay vòng dài khoảng 260°, quấn dây và kẹt motor. Cách sửa: giới hạn phần mềm ±147° và bài tự kiểm cho từng khớp chạy 12°.
- Cáp USB kém là lỗi phần cứng số một.

### Đánh giá cho nhóm

- **★ OpenArm v1.0 — nên lấy gần như nguyên ý tưởng:** bộ lọc One Euro + vùng chết + bỏ bước nhảy theo từng khớp; độ tin cậy từng khớp; sửa đảo nhãn trái/phải; tư thế khoá thay cho hiệu chuẩn; dead-man, giới hạn bước, trộn mềm khi bắt đầu; máy trạng thái HOLD.
- **★ Hợp với laptop nhóm:** MediaPipe chạy trong trình duyệt dùng GPU qua WebGL, nên tận dụng được GPU Intel tích hợp; không cần NVIDIA.
- **Phải viết lại cho 7 khớp:** SO-101 là tay phẳng có đế xoay. `reachJoints` giải IK 2 khâu trong một mặt phẳng; OpenArm có vai 3 bậc (J1–J3) nên phải tách góc theo thứ tự nâng trước → dang → xoay, hoặc giải IK 7 khớp trên URDF v1.0.
- **Thay bridge:** `SO101Follower.send_action()` → `openarm_follower` của LeRobot (cũng nhận góc theo độ) hoặc `openarm_can` trực tiếp. Bước 2.5°/tick ở 30 Hz nên hạ xuống cho khớp vai OpenArm nặng hơn.
- **Bản quyền:** không có license chung, nên viết lại theo ý tưởng thay vì chép code.

## Repo 4 — YashVG/kinetic-arm-lab

Repo này chỉ là mô phỏng trong trình duyệt, tay ảo 3 khớp. Giá trị của nó nằm ở code được viết sạch, có test, và cách xử lý "ly hợp" (clutch) cùng mất tracking rất rõ ràng. [Link repo](https://github.com/YashVG/kinetic-arm-lab), commit 5b6fc21 ngày 06/09/2026 (cả repo chỉ có 1 commit).

| Mục | Nội dung |
| --- | --- |
| Nền tảng | React + Vite + TypeScript, Three.js; Node.js 22.13+, Chrome/Edge có WebGL |
| Mô hình | MediaPipe Tasks Vision 1.0.1, Pose Landmarker **Lite** (float16), chạy trong Web Worker |
| Điểm cần thấy | 2 vai (11, 12), khuỷu phải (14), cổ tay phải (16) |
| Robot | Tay ảo 3 khớp (xoay đế, vai, khuỷu), cánh tay 0.43 m, cẳng tay 0.38 m; không nối robot thật |
| Kiểm thử | `npm test`: hiệu chuẩn thiếu hông, biến đổi toạ độ, mất tracking, giới hạn tốc độ, IK trên 1.331 điểm đích |
| License | Không có file license; các thư viện bên trong (MediaPipe, Three.js) ghi rõ license trong `THIRD_PARTY.md` |

### Cách hoạt động (`lib/controller.ts`, `lib/kinematics.ts`)

1. **Khung toạ độ theo thân:** dùng điểm 3D thế giới của MediaPipe. Trục x = đường nối vai trái→vai phải; trục y = hướng "lên" của camera chiếu vuông góc với x; z = x × y.
2. **Vị trí cổ tay:** cổ tay phải trừ vai phải, chiếu lên 3 trục, rồi chia cho bề rộng vai. Nhờ vậy không phụ thuộc người cao hay thấp.
3. **Hiệu chuẩn tư thế trung tính:** gom 30 khung, nếu độ dao động quá 0.09 bề rộng vai thì bắt giữ lại tay. Trung bình 30 khung là điểm gốc.
4. **Ly hợp (engage):** mỗi lần bấm engage, điểm gốc của người và điểm gốc của robot được đặt lại theo vị trí hiện tại, nên robot không bao giờ giật. Người dùng có thể nhả ly hợp, đưa tay về chỗ thoải mái, rồi engage lại.
5. **Mục tiêu robot** = điểm gốc robot + 0.4 × (cổ tay hiện tại − điểm gốc người). Lọc mũ với hằng số thời gian 120 ms.
6. **IK giải tích 3 khớp:** kẹp mục tiêu trong hộp làm việc, kéo vào nếu vượt tầm với, tính góc khuỷu bằng định lý cosin, góc vai bằng hai `atan2`, góc đế bằng `atan2(z, x)`.
7. **Giới hạn tốc độ:** mỗi khớp tối đa π/2 rad/s (90°/s).
8. **Mất tracking:** độ tin cậy dưới 0.65 hoặc khung cũ hơn 300 ms thì dừng và **phải engage lại bằng tay**, không tự chạy tiếp.

Phần hiệu chỉnh ống kính (tuỳ chọn): script Python OpenCV đo bàn cờ, xuất ma trận camera + 5 hệ số méo; trình duyệt khử méo ảnh trước khi đưa vào MediaPipe. Tác giả ghi rõ: điều này chưa chứng minh làm tracking chính xác hơn.

### Giới hạn tác giả tự ghi

- Giả sử một người, đứng thẳng, camera đặt ngang. Không theo dõi thân cúi, nên ngả người ra trước cũng làm robot di chuyển.
- Độ sâu từ một camera chỉ là ước lượng; độ tin cậy (visibility) không đo độ chính xác vị trí.
- Test dùng dữ liệu tổng hợp; **độ chính xác, độ trễ với webcam thật chưa được đo**.

### Đánh giá cho nhóm

- **★ OpenArm v1.0 — nên lấy:** hàm `bodyWrist` (chuẩn hoá cổ tay theo bề rộng vai trong khung thân), cơ chế ly hợp đặt lại gốc mỗi lần engage, quy tắc mất tracking phải engage lại bằng tay, cách viết test cho bộ điều khiển bằng dữ liệu giả.
- **Không dùng được:** IK 3 khớp và tay ảo; không có hướng cổ tay, không có bàn tay, không có kẹp.
- **Khác ý tưởng của nhóm:** repo điều khiển theo **vị trí cổ tay** (task-space), không chép góc khớp. Phù hợp nếu nhóm chọn cách 2 (IK trên URDF).

## Repo 5 — perfanalytics/pose2sim

Pose2Sim là công cụ mocap không marker **chạy offline** trên video từ 2 camera trở lên, ra góc khớp 3D chuẩn y sinh. Nó không dùng được cho teleop thời gian thực, nhưng rất hợp để làm ground truth và để thu động tác Thái Cực từ video. [Link repo](https://github.com/perfanalytics/pose2sim), commit 3eda69f ngày 27/09/2026.

| Mục | Nội dung |
| --- | --- |
| Nền tảng | Python ≥ 3.11, cài bằng `uv`; OpenSim; RTMPose qua `rtmlib` (ONNX Runtime, chạy được CPU) |
| Đầu vào | Video từ **ít nhất 2 camera** bất kỳ (điện thoại, webcam, GoPro) |
| Đầu ra | Quỹ đạo điểm 3D (`.trc`) và góc khớp 3D (`.mot`) sau IK trên mô hình xương OpenSim đã co giãn theo người |
| Độ chính xác tác giả nêu | Sai số góc khớp 2–6°, theo các nghiên cứu đã phản biện |
| License | BSD 3-Clause, được dùng lại code kể cả thương mại, giữ ghi công |

### Pipeline 8 bước

1. `calibration()`: tính nội tham (bàn cờ, một lần cho mỗi camera) và ngoại tham (mỗi lần dời camera), hoặc nhập từ AniPose, FreeMoCap, Caliscope…
2. `poseEstimation()`: ước lượng điểm 2D, mặc định RTMPose; có chế độ `lightweight` nhanh hơn.
3. `synchronization()`: đồng bộ thời gian giữa các camera bằng tương quan vận tốc dọc của các điểm; người quay cần làm một động tác dọc dứt khoát.
4. `personAssociation()`: ghép cùng một người giữa các camera (bỏ qua nếu chỉ có 1 người).
5. `triangulation()`: tam giác hoá có trọng số theo độ tin cậy; loại dần camera có sai số chiếu lại lớn; nội suy lỗ hổng ngắn.
6. `filtering()`: Butterworth, Kalman, One Euro, GCV spline, LOESS, Gaussian, trung vị; tuỳ chọn Hampel để loại điểm lạ.
7. `markerAugmentation()`: mạng LSTM của Stanford ước lượng 47 marker ảo; có ích khi ít hơn 4 camera nhưng có thể kém chính xác hơn.
8. `kinematics()`: co giãn mô hình OpenSim theo chiều cao người và giải IK ra góc khớp.

### ★ Điểm đáng chú ý cho OpenArm v1.0

Mô hình OpenSim của Pose2Sim có **7 bậc tự do mỗi tay**: `arm_flex`, `arm_add`, `arm_rot` (vai), `elbow_flex`, `pro_sup` (xoay cẳng tay), `wrist_flex`, `wrist_dev` (cổ tay). Mình đọc thấy tên này trong file `Model_Pose2Sim_simple.osim`. Cấu trúc đó khớp với chuỗi OpenArm:

| Khớp OpenArm | Chuyển động | Tọa độ OpenSim tương ứng (dự kiến) |
| --- | --- | --- |
| J1 | nâng tay ra trước | `arm_flex` |
| J2 | dang tay ngang | `arm_add` |
| J3 | xoay cánh tay | `arm_rot` |
| J4 | gập khuỷu | `elbow_flex` |
| J5 | xoay cẳng tay | `pro_sup` |
| J6 | cổ tay (trục x) | `wrist_dev` |
| J7 | cổ tay (trục y) | `wrist_flex` |

Bảng này là suy luận của mình từ hướng trục. Dấu và điểm 0 của từng cặp phải kiểm tra bằng thực nghiệm. Ngoài ra, OpenSim dùng thứ tự xoay vai khác OpenArm và chế độ `use_simple_model` biến vai thành khớp cầu, nên ở góc lớn giá trị có thể lệch nhau.

### Đánh giá cho nhóm

- **Không dùng cho teleop trực tiếp:** xử lý theo lô sau khi quay xong. Bản demo chạy mất 1–2 phút trên laptop, tinh chỉnh còn vài giây. Tác giả gợi ý dùng [Sports2D](https://github.com/davidpagnon/Sports2D) nếu cần thời gian thực với 1 camera, nhưng khi đó động tác phải nằm trong mặt phẳng dọc hoặc ngang.
- **★ Dùng làm ground truth:** quay cùng lúc bằng điện thoại + laptop, chạy Pose2Sim offline, rồi so với góc mà pipeline MediaPipe thời gian thực của nhóm tính ra. Đây là cách đo sai số không cần hệ mocap đắt tiền.
- **★ Dùng cho nhánh Thái Cực:** quay một người tập bằng 2 camera → `.mot` có 7 góc mỗi tay → ánh xạ theo bảng trên → phát lại trên OpenArm.
- **Nên lấy code:** calibration, triangulation có trọng số, các bộ lọc. License BSD cho phép dùng lại.
- **Lưu ý khi quay:** tác giả khuyên 2 camera, một ở phía trước, một lệch 45°, cả hai ngang hông; tốc độ khung dưới 60 fps làm giảm độ chính xác với động tác nhanh.

## Đề xuất: ghép thành pipeline cho nhóm

Nên lấy phần đo và lọc của repo 3, cơ chế ly hợp của repo 4, tự viết khâu ánh xạ 7 khớp cho OpenArm, và dùng Pose2Sim để đo sai số.

```
 THỊ GIÁC (trình duyệt hoặc Python)                          
 ┌────────────────────┐   ┌──────────────────────┐   ┌──────────────────────────┐
 │ Camera 2D +        │──►│ Đặc trưng theo khung │──►│ Lọc One Euro + vùng chết │
 │ MediaPipe Pose/Hand│   │ thân (MakeFrame)     │   │ + độ tin cậy từng khớp   │
 │ (repo 3, repo 4)   │   │ (SEW-Mimic)          │   │ (repo 3)                 │
 └────────────────────┘   └──────────────────────┘   └────────────┬─────────────┘
                                                                  │
 ĐIỀU KHIỂN (Python trên laptop)                                  ▼
 ┌────────────────────┐   ┌──────────────────────┐   ┌──────────────────────────┐
 │ OpenArm v1.0       │◄──│ Bridge an toàn 30 Hz │◄──│ Ánh xạ 7 khớp (tự viết,  │
 │ openarm_can CAN-FD │   │ dead-man, ly hợp,    │   │ SP1/SP2 kiểu SEW-Mimic)  │
 │                    │   │ giới hạn (repo 3, 4) │   │                          │
 └────────────────────┘   └──────────────────────┘   └──────────────────────────┘

 ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
   Offline: 2 camera → Pose2Sim (repo 5) → góc tham chiếu, so sai số
 └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
```

> Trong repo `openarm_shadow` cả phần thị giác lẫn điều khiển đều chạy bằng Python (MediaPipe Tasks trên CPU), nên không cần WebSocket; phương án trình duyệt để dành khi CPU không đủ fps.

Đọc theo mũi tên: hàng trên là phần thị giác, hàng dưới là phần điều khiển; ô nét đứt chạy riêng, sau khi quay xong.

Thứ tự làm:

1. Chạy MediaPipe Pose + Hand trong trình duyệt (`@mediapipe/tasks-vision@0.10.21` như repo 3), vẽ khung xương lên ảnh, in đặc trưng. Chưa nối robot.
2. Viết ánh xạ J1–J4 từ vector vai→khuỷu và mặt phẳng khuỷu, theo thứ tự trục nâng trước → dang → xoay. Kiểm tra trên viewer MuJoCo v1.
3. Thêm lọc One Euro, vùng chết, độ tin cậy từng khớp (repo 3) và ly hợp đặt lại gốc (repo 4).
4. Viết bridge Python: nhận góc qua WebSocket, gửi xuống `openarm_can`; dead-man 0.4 s, giới hạn bước, trộn mềm khi bắt đầu. Thử trên robot thật từng khớp một.
5. Thêm J5–J7 và kẹp từ Hand Landmarker.
6. Đo sai số: quay cùng lúc bằng điện thoại + laptop, chạy Pose2Sim offline, so với góc thời gian thực.

Nếu nhóm muốn đi theo ROS 2 thay vì gọi `openarm_can` trực tiếp, repo 1 cho sẵn cách dựng OpenArm v10 với MoveIt Servo; con đường này nặng hơn và repo đó viết cho Ubuntu 22.04.

## Bài báo 1: SEW-Mimic

Bản dịch tóm tắt và mục I–III (phần sau chưa lấy được bản gốc): [`dich_sew_mimic.md`](dich_sew_mimic.md).

## Bài báo 2: Vision-Based Hand Shadowing

Bản dịch tóm tắt và mục I–IV (phần V trở đi chưa lấy được bản gốc): [`dich_hand_shadowing.md`](dich_hand_shadowing.md).
