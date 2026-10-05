# Kiến trúc hệ thống openarm_shadow

> Tài liệu mô tả **hệ thống đang chạy thật** (trạng thái code + config ngày 03/10/2026): kiến trúc, các luồng,
> thuật toán, tham số, số đo hiệu năng, giới hạn đã biết và các hướng cải tiến. Mục tiêu: đọc xong nắm được dự án
> hoạt động thế nào, đủ cơ sở để đánh giá và quyết định thay đổi.
>
> Mọi số liệu hiệu năng lấy từ `data/2026-10-03_hardware_baseline/`. Đường dẫn file tính từ gốc repo.

---

## Mục lục

1. [Tổng quan](#1-tổng-quan)
2. [Phần cứng và môi trường hiện tại](#2-phần-cứng-và-môi-trường-hiện-tại)
3. [Kiến trúc tổng thể](#3-kiến-trúc-tổng-thể)
4. [Mô hình luồng (thread) và đồng bộ](#4-mô-hình-luồng-thread-và-đồng-bộ)
5. [Hành trình của một khung hình](#5-hành-trình-của-một-khung-hình)
6. [Hệ toạ độ và quy ước](#6-hệ-toạ-độ-và-quy-ước)
7. [Perception: từ ảnh tới quan sát tay người](#7-perception-từ-ảnh-tới-quan-sát-tay-người)
8. [Fusion nhiều camera](#8-fusion-nhiều-camera)
9. [Pipeline: độ tin cậy, lọc điểm, retarget, lọc khớp](#9-pipeline-độ-tin-cậy-lọc-điểm-retarget-lọc-khớp)
10. [Retarget kiểu SEW-Mimic](#10-retarget-kiểu-sew-mimic)
11. [SafetyGate](#11-safetygate)
12. [Backend robot (CAN-FD)](#12-backend-robot-can-fd)
13. [Hiệu chuẩn](#13-hiệu-chuẩn)
14. [Máy trạng thái vận hành](#14-máy-trạng-thái-vận-hành)
15. [Cấu hình](#15-cấu-hình)
16. [Công cụ đo, ghi và phân tích](#16-công-cụ-đo-ghi-và-phân-tích)
17. [Hiệu năng đo được và phân tích độ trễ](#17-hiệu-năng-đo-được-và-phân-tích-độ-trễ)
18. [Các lớp an toàn](#18-các-lớp-an-toàn)
19. [Giới hạn đã biết, rủi ro, nợ kỹ thuật](#19-giới-hạn-đã-biết-rủi-ro-nợ-kỹ-thuật)
20. [Hướng cải tiến đề xuất](#20-hướng-cải-tiến-đề-xuất)
21. [Bảng tra cứu tham số](#21-bảng-tra-cứu-tham-số)
22. [Kiểm thử](#22-kiểm-thử)

---

## 1. Tổng quan

**Bài toán.** Teleop hai tay robot OpenArm v1.0 (mỗi tay 7 khớp + kẹp) bằng camera, không cần găng tay hay
marker: người đứng trước camera, robot bắt chước dáng tay (shadowing) theo thời gian thực.

**Ý tưởng cốt lõi.**

- Nhận diện người bằng **MediaPipe Pose + Hand** (CPU) trên nhiều camera.
- Hợp nhất 3D bằng **triangulation có trọng số** (+ depth RealSense làm bằng chứng phụ).
- Ánh xạ sang khớp robot bằng cách **căn hướng** (không căn vị trí) từng đoạn tay: cánh tay trên, cẳng tay,
  bàn tay, theo ý tưởng SEW-Mimic, giải dạng đóng bằng các bài toán con SP1/SP2. Tay người dài hay ngắn
  không ảnh hưởng.
- Mỗi khớp có **độ tin cậy riêng**: khớp nào nhìn kém thì riêng khớp đó đứng yên.
- Mọi lệnh đi qua **SafetyGate** (giới hạn góc/vận tốc, ly hợp, dead-man, chống hai tay va nhau) rồi mới tới
  motor qua **CAN-FD, điều khiển MIT** (kp/kd + vị trí + vận tốc + mô-men bù).

**Trạng thái (03/10/2026).**

- Chạy thật **cả hai tay J1–J7 + kẹp**, 3 camera (webcam laptop + D455 + D435i), chế độ triangulate,
  tự engage sau khi giữ READY 2 s. Lần chạy dài nhất ghi lại: 136 s (`run_grasp_drop.npz`).
- Perception thực tế **12,5 fps**, vòng điều khiển **86 Hz** (đặt 100 Hz).
- Vấn đề chính còn lại: **độ trễ cảm nhận lớn** (mục tiêu → robot 245–595 ms) và **bước nhảy lớn ở J3/J5**
  (13 lần tay phải, 19 lần tay trái trong 136 s). Xem [mục 17](#17-hiệu-năng-đo-được-và-phân-tích-độ-trễ).

**Ngoài phạm vi:** điều khiển theo vị trí đầu kẹp (IK vị trí), tránh va chạm tay–thân/bàn, học bắt chước
(chỉ mới có công cụ ghi dữ liệu).

---

## 2. Phần cứng và môi trường hiện tại

| Thành phần | Chi tiết |
| --- | --- |
| Máy tính | Intel Core Ultra 5 125H (18 luồng), Ubuntu 24.04, kernel 7.0, không dùng GPU rời |
| Python | 3.12, venv `--system-site-packages` (để thấy `python3-openarm-can` cài bằng apt) |
| Camera `front` | Webcam laptop tích hợp, chỉ số 0, 1280×720 MJPG 30 fps, trực diện. **Camera tham chiếu** (khung thế giới) |
| Camera `side_right` | Intel RealSense **D455** (serial 341522301338), USB 3.2, 848×480 @30, RGB + depth, lệch 45–60° phía tay phải |
| Camera `side_left` | Intel RealSense **D435i** (serial 243122071323), USB 3.2, 848×480 @30, RGB + depth, lệch 45–60° phía tay trái |
| CAN | PEAK PCAN-USB Pro FD → `can0` (tay phải), `can1` (tay trái); 1 Mbit/s arbitration, 5 Mbit/s data |
| Robot | OpenArm v1.0, mỗi tay: J1–J2 DM8009, J3–J4 DM4340, J5–J7 DM4310, kẹp DM4310 (ID gửi 0x01–0x08, nhận 0x11–0x18) |
| Model MediaPipe | `pose_landmarker_full.task`, `hand_landmarker.task` (có sẵn `pose_landmarker_lite.task` chưa dùng) |
| Hiệu chuẩn camera | ChArUco 7×5, ô 40 mm, marker 30 mm, `DICT_5X5_1000`; RMS hiện tại 0,52 px và 0,64 px |

---

## 3. Kiến trúc tổng thể

### 3.1 Sơ đồ khối

```
 ┌────────────── NGUỒN ẢNH (cameras/sources.py, cameras/multicam.py) ───────────────┐
 │  webcam front (OpenCV/V4L2)   D455 (RGB+depth)   D435i (RGB+depth)               │
 │  mỗi camera 1 luồng đọc → bộ đệm 12 khung có timestamp → ghép khung theo thời gian │
 └───────────────────────────────────────┬──────────────────────────────────────────┘
                                          │ MultiSample (3 CameraSample + t + skew + stale)
 ┌──────────────── PERCEPTION (perception/, fusion/) ───────────────────────────────┐
 │  mỗi camera: MediaPipe Pose (≤2 người) ∥ Hand (≤2 bàn tay) → khoá người điều khiển │
 │  → gán bàn tay theo cổ tay Pose → quan sát 2D (+ depth nhấc lên 3D)               │
 │  hợp nhất: triangulate có trọng số vai/khuỷu/cổ tay/hông + 21 điểm mỗi bàn tay     │
 │  → khung thân → lọc đốt bàn tay → Kabsch lòng bàn tay → OrientationFusion          │
 └───────────────────────────────────────┬──────────────────────────────────────────┘
                                          │ Frame{arms: {right/left: ArmObs(s,e,w,H,grip,conf)}}
 ┌──────────────── PIPELINE (retarget/, filtering/) ────────────────────────────────┐
 │  direct/mirror → độ tin cậy từng nhóm khớp → ArmShape (độ dài xương) → Kalman điểm │
 │  → retarget SEW (SP2/SP1) → độ tin cậy J3 theo góc khuỷu → JointFilter 8 phần tử   │
 │  (One Euro + vùng chết + xác nhận bước nhảy) ; kẹp: GripMapper                     │
 └───────────────────────────────────────┬──────────────────────────────────────────┘
                                          │ targets[side] = 8 số (7 góc URDF rad + kẹp 0..1, NaN = giữ)
 ┌──────────────── SAFETYGATE (safety/gate.py) — chạy trong luồng điều khiển 100 Hz ┐
 │  ly hợp + ramp smoothstep → dead-man → kẹp giới hạn mềm → giới hạn vận tốc         │
 │  (+ ramp riêng khớp vừa chạy lại) → [tuỳ chọn bám vận tốc] → chống va chạm 2 tay   │
 └───────────────────────────────────────┬──────────────────────────────────────────┘
                                          │ cmd[side] (8 số), dq[side] (tuỳ chọn)
 ┌──────────────── ROBOT (robot/openarm_can_robot.py | robot/sim.py) ───────────────┐
 │  chốt giới hạn motor → URDF→motor (sign, offset, wrap gần góc đo) → MIT(kp·ramp,  │
 │  kd, q, dq, τ_gravity) → đọc phản hồi → lọc số đọc rác → kiểm tra phản hồi mới    │
 └──────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Bản đồ module

Thư viện chia theo nhiệm vụ (tách ngày 05/10/2026, không đổi thuật toán: đầu ra trên dữ liệu thô giống hệt trước khi
tách). Phụ thuộc chỉ đi một chiều: `core` ← `perception` ← `fusion`; `core` + `filtering` ← `retarget`;
`runtime` nối tất cả. `cameras`, `perception`, `fusion`, `retarget`, `filtering`, `safety` không phụ thuộc robot hay
giao diện, nên dùng lại được trong node ROS 2 hoặc công cụ offline.

| Package / file | Dòng | Vai trò |
| --- | ---: | --- |
| `scripts/shadow.py` | 50 | Điểm vào: đọc tham số dòng lệnh, ghép config, gọi `runtime.run` |
| `config.py` | 32 | Đọc `default.yaml` rồi ghép đè lần lượt các file `--config` |
| **core/** | | **Toán dùng chung** |
| `core/geometry.py` | 147 | Rodrigues, SP1/SP2/SP4, `make_frame`, khoảng cách đoạn–đoạn |
| `core/rotations.py` | 59 | Ma trận ↔ quaternion, SLERP, khoảng cách góc |
| `core/kinematics.py` + `core/data/` | 115 | FK OpenArm v1.0 từ JSON sinh từ URDF |
| **cameras/** | | **Nguồn ảnh** |
| `cameras/sources.py` | 176 | `OpenCVSource` (webcam/video/URL), `RealSenseSource` (RGB+depth align, lọc disparity) |
| `cameras/multicam.py` | 184 | `MultiCameraSource`: mỗi camera 1 luồng đọc, ghép khung theo thời điểm (`MultiSample`) |
| `cameras/raw_replay.py` | 134 | `RawReplaySource`: phát lại dữ liệu thô đã ghi, cùng giao diện `MultiCameraSource` |
| `cameras/calibration.py` | 128 | ChArUco: tư thế bảng, ngoại tham số tương đối, nội tham số webcam, đo trễ giữa camera |
| **perception/** | | **Nhận diện trên một camera** |
| `perception/types.py` | 55 | Chỉ số điểm mốc, `ArmObs`, `Frame` (kiểu dữ liệu dùng chung mọi khâu) |
| `perception/landmarker.py` | 391 | `Perception`: MediaPipe Pose ∥ Hand, lịch chạy Pose, đường 1 camera + depth D455, ổn định hướng |
| `perception/operator.py` | 42 | Khoá người điều khiển (`select_operator`) |
| `perception/hand_assign.py` | 23 | Gán bàn tay vào cổ tay Pose |
| `perception/depth.py` | 203 | Depth tại điểm mốc, chiếu/khử chiếu, 21 điểm bàn tay metric, mặt phẳng lòng bàn tay |
| `perception/hand_geometry.py` | 52 | Đếm ngón xoè, khung lòng bàn tay |
| `perception/body.py` | 16 | Khung thân từ vai + hông |
| **fusion/** | | **Hợp nhất nhiều camera** |
| `fusion/camera_model.py` | 118 | `CameraModel`, đọc file hiệu chuẩn, kiểm tra độ phân giải |
| `fusion/triangulation.py` | 125 | `triangulate_weighted`, `fuse_point`, sai số chiếu lại, hệ số tin cậy theo sai số |
| `fusion/person_match.py` | 126 | `OperatorMatcher`: cùng một người ở mọi camera |
| `fusion/hand.py` | 209 | `HandFuser`: gác cổng theo cổ tay, triangulate 21 điểm, Kabsch, kẹp, hướng tay nhiều nguồn |
| `fusion/hand_model.py` | 115 | `HandShape` (độ dài đốt), `PalmModel` (Kabsch với khuôn), kiểm tra giải phẫu |
| `fusion/orientation.py` | 116 | `OrientationFusion` (máy trạng thái hướng tay, chống lật) |
| `fusion/multiview.py` | 299 | `MultiViewPerception`: quan sát từng camera, hợp nhất thân, điều phối các khối trên |
| **filtering/** | | **Lọc nhiễu** |
| `filtering/joint.py` | 97 | `OneEuro`, `JointFilter` (vùng chết, xác nhận bước nhảy, độ tin cậy từng khớp) |
| `filtering/points.py` | 106 | `EMA`, `ArmShape`, `PointKalman` |
| **retarget/** | | **Ánh xạ người → robot** |
| `retarget/sew.py` | 156 | Ánh xạ hướng chi → 7 góc khớp (SP2/SP1), chọn nghiệm, phản chiếu gương |
| `retarget/grip.py` | 86 | Tỉ số ngón cái–trỏ → độ mở kẹp, hiệu chuẩn theo người |
| `retarget/pipeline.py` | 260 | `ShadowPipeline`: độ tin cậy → lọc điểm → retarget → lọc khớp; tự hiệu chuẩn hướng tay |
| **safety/** | | |
| `safety/gate.py` | 234 | `SafetyGate` |
| **robot/** | | **Backend robot** |
| `robot/openarm_can_robot.py` | 326 | Backend CAN thật |
| `robot/sim.py` | 30 | Robot lý tưởng: đo = lệnh |
| `robot/gravity.py` | 31 | Bù trọng lực Pinocchio (tắt, chưa kiểm chứng) |
| **runtime/** | | **Chạy teleop** |
| `runtime/app.py` | 247 | `open_perception`, `run`: nối các khối, phím, hiển thị, về tư thế nghỉ khi thoát |
| `runtime/controller.py` | 77 | Luồng điều khiển 100 Hz, `park`, chặn engage khi cổ tay chưa hiệu chuẩn |
| `runtime/worker.py` | 49 | Luồng perception, luôn giữ kết quả mới nhất |
| `runtime/session.py` | 46 | `AutoEngage`: tự engage sau khi giữ READY liên tục |
| `runtime/recorder.py` | 50 | `--record` |
| **viz/** | | **Hiển thị** |
| `viz/draw.py` | 145 | Vẽ khung xương lên ảnh, hình que robot hai góc nhìn |
| `viz/hud.py` | 141 | Dòng chữ chẩn đoán, huy hiệu READY/FOLLOW |

--- | ---: | --- |
| `scripts/shadow.py` | 50 | Điểm vào: đọc tham số dòng lệnh, ghép config, gọi `app.run` |
| `openarm_shadow/runtime/app.py` | 520 | Vòng chạy chính, luồng điều khiển, luồng perception, phím, hiển thị, ghi `--record`, về tư thế nghỉ |
| `openarm_shadow/config.py` | 27 | Đọc `default.yaml` rồi ghép đè lần lượt các file `--config` |
| `openarm_shadow/cameras/sources.py` | 176 | `OpenCVSource` (webcam/video/URL), `RealSenseSource` (RGB+depth align, lọc disparity) |
| `openarm_shadow/perception/landmarker.py` | 773 | MediaPipe 1 camera, khoá người, gán bàn tay, depth D455, khung lòng bàn tay, ổn định hướng |
| `openarm_shadow/fusion/multiview.py` | 1024 | Đồng bộ nhiều camera, mô hình camera, `fuse_point`, khoá cùng người giữa các camera, hợp nhất thân + bàn tay |
| `openarm_shadow/fusion/hand.py` | 263 | Gán bàn tay–cổ tay, `HandShape`, `PalmModel` (Kabsch), `OrientationFusion` (máy trạng thái hướng tay) |
| `openarm_shadow/retarget/pipeline.py` | 259 | Nối perception → retarget → lọc; tự hiệu chuẩn hướng tay; kẹp |
| `openarm_shadow/retarget/sew.py` | 156 | Ánh xạ hướng chi → 7 góc khớp (SP2/SP1), chọn nghiệm, phản chiếu gương |
| `openarm_shadow/core/geometry.py` | 147 | Rodrigues, SP1/SP2/SP4, `make_frame`, khoảng cách đoạn–đoạn |
| `openarm_shadow/core/kinematics.py` | 115 | FK OpenArm v1.0 từ JSON sinh từ URDF |
| `openarm_shadow/filtering/joint.py` | 197 | `OneEuro`, `JointFilter`, `EMA`, `ArmShape`, `PointKalman` |
| `openarm_shadow/retarget/grip.py` | 86 | Tỉ số ngón cái–trỏ → độ mở kẹp (liên tục / theo mức), hiệu chuẩn theo người |
| `openarm_shadow/safety/gate.py` | 234 | `SafetyGate` |
| `openarm_shadow/robot/openarm_can_robot.py` | 326 | Backend CAN thật |
| `openarm_shadow/robot/sim.py` | 30 | Robot lý tưởng: đo = lệnh |
| `openarm_shadow/robot/gravity.py` | 31 | Bù trọng lực Pinocchio (tắt, chưa kiểm chứng) |
| `openarm_shadow/cameras/calibration.py` | 128 | ChArUco: tư thế bảng, ngoại tham số tương đối, nội tham số webcam, đo trễ giữa camera |
| `openarm_shadow/viz/draw.py` | 145 | Vẽ khung xương lên ảnh, hình que robot hai góc nhìn |

---

## 4. Mô hình luồng (thread) và đồng bộ

Khi chạy `--source multi` với 3 camera, tiến trình có các luồng sau:

```
 Luồng đọc camera ×3 (MultiCameraSource._loop)       ← blocking read từng camera, gắn t = monotonic − latency_s
        │  deque(12) mỗi camera, Condition
        ▼
 Luồng perception (app.PerceptionWorker)              ← cap.read() ghép khung → MultiViewPerception.process
        │   ├─ ThreadPool(3): mỗi camera 1 Perception.process
        │   │     └─ ThreadPool(1) "hand": Hand chạy song song với Pose trên cùng ảnh
        │   └─ hợp nhất (tuần tự, Python)
        │  chỉ giữ kết quả MỚI NHẤT (luồng chính chậm thì bỏ khung cũ)
        ▼
 Luồng chính (app.run)                                ← auto-calib, auto-engage, pipeline.step, set_target,
        │                                                ghi log, vẽ UI (update_hz), đọc phím
        │  ctl.lock
        ▼
 Luồng điều khiển (app.Controller, control_hz=100)    ← gate.step(dt) → robot.send(cmd, dq) → [trace]
```

Điểm cần biết:

- **Hai nhịp tách rời.** Perception ~12,5 Hz đặt *mục tiêu*; luồng điều khiển ~86–100 Hz tiến *lệnh* về mục
  tiêu theo giới hạn vận tốc. Robot không chờ camera.
- **Khoá dùng chung** `ctl.lock` bảo vệ `SafetyGate` (luồng chính gọi `set_target`, `engage`, `disengage`; luồng
  điều khiển gọi `step`). `robot.send` gọi ngoài khoá.
- **MediaPipe nhả GIL** nên chạy song song thật; phần hợp nhất/lọc/retarget là Python thuần, tranh GIL với luồng
  điều khiển. Đây là lý do khả dĩ nhất khiến vòng điều khiển chỉ đạt 86 Hz (chưa đo riêng).
- **Lỗi phần cứng** trong luồng điều khiển → `ctl.error`, `running=False` → luồng chính thoát vòng, khối `finally`
  về tư thế nghỉ (nếu luồng điều khiển không lỗi) rồi `relax()` + tắt motor.
- **Mất camera**: `PerceptionWorker.get()` chờ tối đa 10 s; quá hạn → dừng chương trình. Trong lúc chờ,
  dead-man (0,4 s) đã cho robot đứng yên.

---

## 5. Hành trình của một khung hình

Từ lúc ảnh được chụp tới lúc motor nhận lệnh (chế độ triangulate 3 camera, config đang dùng):

| # | Bước | Nơi | Dữ liệu ra |
| --- | --- | --- | --- |
| 1 | Mỗi camera đọc khung, gắn `t = monotonic() − latency_s` | `MultiCameraSource._loop` | `(t, CameraSample)` vào deque |
| 2 | Lấy khung `front` mới nhất (sync `latest`), mỗi camera phụ lấy khung có `t` gần nhất; lệch > `max_skew_s` (40 ms) → camera đó `stale` khung này | `MultiCameraSource.read` | `MultiSample` |
| 3 | Mỗi camera (không stale): MediaPipe Pose (mỗi 2 khung, `pose_interval: 2`) ∥ Hand; chọn người điều khiển; gán bàn tay vào cổ tay | `Perception.process` | `Frame` 2D từng camera |
| 4 | Quan sát từng camera: chuẩn hoá điểm (khử méo), nhấc depth thành điểm 3D thế giới | `_view_obs` | dict pose/hand |
| 5 | Triangulate 8 điểm thân (vai, khuỷu, cổ tay, hông) bằng `fuse_point`; độ tin cậy × hệ số sai số chiếu lại | `fuse` | `W[33×3]`, `vis` |
| 6 | Khung thân từ 2 vai + giữa hông; làm mượt SLERP, chặn nhảy > 45° | `body_frame` | `R_body`, gốc |
| 7 | Bàn tay: lọc camera xa cổ tay → triangulate 21 điểm → `HandShape` → `PalmModel` (Kabsch) → kiểm tra góc bàn tay–cẳng tay → `OrientationFusion` | `_fuse_hand` | `H`, `grip`, `conf.hand`, `conf.grip` |
| 8 | Trộn cổ tay Pose với cổ tay Hand (70 %) nếu cách < 8 cm | `fuse` | `ArmObs(s,e,w,H,…)` khung thân |
| 9 | Auto-calib hướng tay trung tính (khi chưa engage) và auto-engage | `app.run`, `pipeline.auto_calibrate_hand_neutral` | |
| 10 | Độ tin cậy 3 nhóm khớp; `ArmShape`; `PointKalman` (s,e,w) | `ShadowPipeline.step` | điểm đã lọc |
| 11 | Retarget: `u = e−s`, `l = w−e`, `H` → `q` (7 góc) | `ArmRetargeter.solve` | `q`, `RetargetInfo` |
| 12 | Độ tin cậy J3 theo góc gập khuỷu; `JointFilter` 8 phần tử | `JointFilter` | `targets[side]`, `held` |
| 13 | `gate.set_target(targets, fresh, t_frame, held)` | luồng chính | |
| 14 | Mỗi 10 ms: `gate.step` → `robot.send` → MIT + đọc phản hồi + lọc rác | luồng điều khiển | lệnh motor |

---

## 6. Hệ toạ độ và quy ước

| Khung | Trục | Ghi chú |
| --- | --- | --- |
| Camera (OpenCV/RealSense) | x phải, y xuống, z ra xa camera | Điểm MediaPipe "world" cũng theo hướng trục camera (gốc ở hông/cổ tay) |
| Thế giới fusion | = khung camera tham chiếu `front` | `X_cam = R·X_world + t` cho mỗi camera, từ hiệu chuẩn ChArUco |
| Thân người | x **trước**, y **trái**, z **lên**; gốc giữa hai vai | `make_frame(vai trái, vai phải, giữa hông)`; thiếu hông thì dùng "lên" của camera |
| Robot (URDF world) | x trước, y trái, z lên | Trùng quy ước thân người → hướng chi người đưa thẳng vào retarget |
| Bàn tay `H` | cột x = hướng ngón (cổ tay → giữa gốc ngón giữa/áp út), y = út → trỏ, z = pháp tuyến **ra khỏi lòng bàn tay** | Tay phải đổi dấu z để hai tay cùng quy ước (`palm_frame_from_depth`) |
| Góc khớp | Rad theo **URDF**; motor = `sign · URDF + offset` (wrap ±180° gần góc đang đo) | `robot.urdf_to_motor` |
| Kẹp | 0 = đóng, 1 = mở hẳn; motor = `closed + f·(open − closed)` | |

**Direct / mirror.** `direct`: tay phải người → tay phải robot (người đứng "trong" robot). `mirror`: tay phải người →
tay trái robot, mọi vector phản chiếu qua mặt phẳng dọc giữa thân (`y → −y`), ma trận bàn tay đổi dấu cột z
(`mirror_rotation`).

---

## 7. Perception: từ ảnh tới quan sát tay người

### 7.1 Nguồn ảnh (`sources.py`)

- **Webcam** (`OpenCVSource`): mở bằng V4L2, buffer 2 khung (1 làm mất nửa fps trên webcam Sonix), thử lần lượt
  các thứ tự đặt định dạng/kích thước/fps và kiểm tra bằng một khung thật; đặt control V4L2 qua `v4l2-ctl`
  (đang dùng: `power_line_frequency=1`, `exposure_dynamic_framerate=0` để không tụt fps khi tối).
- **RealSense** (`RealSenseSource`): stream depth + color cùng độ phân giải, `align` depth → color (depth nằm trong
  pixel RGB), lọc trong miền disparity (spatial — đang **tắt** trong `local_3cam.yaml`, temporal — bật), không
  hole-fill (tránh kéo depth nền vào giữa các ngón). Trả `depth_m` (float32 mét) + intrinsics.

### 7.2 MediaPipe và khoá người điều khiển

- `PoseLandmarker` (`num_poses = 2`) và `HandLandmarker` (`num_hands = 2`), chế độ VIDEO, delegate CPU (XNNPACK).
  Hand chạy ở luồng riêng song song với Pose (~60 → ~43 ms/khung theo ghi chú trong code).
- `pose_interval: 2`: Pose chạy 1/2 khung, khung còn lại dùng lại kết quả; `pose_hold_frames: 6`: Pose hụt ≤ 6 lần
  vẫn giữ kết quả cũ nhưng **hạ visibility × 0,5** để J1–J4 đứng yên thay vì coi điểm cũ là mới.
- **Khoá người** (`select_operator`): lần đầu chọn người có vai rộng nhất × gần giữa ảnh × thấy đủ người; sau đó
  chỉ nhận người có tâm vai cách tâm cũ < `pose_lock_dist` × bề rộng vai và bề rộng vai không đổi quá 2 lần.
  Mất người ≤ 15 lần Pose vẫn giữ khoá. Người khác đi ngang không cướp được quyền.

### 7.3 Gán bàn tay với cổ tay

Không tin nhãn handedness của MediaPipe. `assign_hands_to_wrists`: ghép cặp (bàn tay, cổ tay Pose) theo khoảng
cách tăng dần, mỗi cổ tay ≤ 1 bàn tay, bỏ bàn tay xa mọi cổ tay > 0,35 × bề rộng vai (tay người phía sau).
Chỉ điều khiển 1 tay (`force_hand_side`): nới thêm tới 0,6 × vai nhưng không bao giờ lấy bàn tay đang ở cổ tay
bên kia.

### 7.4 Đường 1 camera (không fusion)

Vẫn dùng được (`--source realsense` hoặc `--source 0`), có giá trị để so sánh/dự phòng:

- **D455 RGB-D**: depth tại landmark lấy median vùng 7×7 quanh pixel, chỉ giữ cụm lệch ≤ 15 cm so với tâm, độ tin
  cậy depth = `√(tỉ lệ hỗ trợ) · e^(−MAD/2cm)`. Đủ depth cho cả vai–khuỷu–cổ tay thì dùng điểm depth (metric),
  thiếu 1 điểm thì dùng trọn bộ MediaPipe world (không trộn hai hệ).
- **21 điểm bàn tay metric** (`fuse_hand_landmarks`): fit robust `z_metric = a·z_mediapipe + b` (bình phương tối thiểu
  có trọng số, loại ngoại lai 3·MAD, làm mượt model 0,7/0,3 giữa khung), trộn với depth đo trực tiếp theo độ tin cậy.
- **Mặt phẳng lòng bàn tay** (`fit_palm_plane`): lấy point cloud trong đa giác 5 điểm lòng bàn tay (đã co mép),
  lọc dải ±6 cm, SVD → pháp tuyến, lặp lại bỏ ngoại lai, RMS > 12 mm thì bỏ. Pháp tuyến này trộn tối đa 85 % với
  pháp tuyến từ landmark.
- **Ổn định hướng** (`_stabilize_orientation`): SLERP với hệ số thích ứng 0,55 → 0,90 khi xoay nhanh, nhảy
  > 155° coi là lật → giữ hướng cũ ≤ 6 khung.
- **Webcam thường**: hướng bàn tay từ điểm world MediaPipe Hand, cùng quy ước khung.

### 7.5 Đếm ngón xoè, tỉ số kẹp

- `open_finger_count`: ngón duỗi nếu `|tip−mcp| / chiều dài chuỗi đốt > 0,78` và `|tip−wrist| / |mcp−wrist| > 1,30`.
  Dùng để: chỉ cập nhật hướng/hiệu chuẩn khi bàn tay xoè.
- Tỉ số kẹp `r = |đầu ngón cái − đầu ngón trỏ| / |gốc ngón giữa − cổ tay|` (không phụ thuộc tay to nhỏ, xa gần).

---

## 8. Fusion nhiều camera

Config đang dùng: `body_source: triangulate` (mọi camera chạy Pose + Hand), `sync: latest`, `pair_wait_s: 0`.

### 8.1 Đồng bộ phần mềm (`MultiCameraSource`)

- Mỗi camera một luồng đọc, bộ đệm 12 khung. Thời điểm khung = thời điểm đến máy − `latency_s` (độ trễ cố định
  của camera, đo bằng `scripts/measure_camera_latency.py`).
- `sync: latest`: lấy khung tham chiếu mới nhất, camera phụ lấy khung gần thời điểm nhất đang có (không chờ vì
  `pair_wait_s: 0`). Lệch > `max_skew_s` = 40 ms → camera đó coi như không có khung lần này (không triangulate khung
  lệch: tay chạy 1 m/s lệch 4 cm).
- `sync: slowest` (không dùng): chờ để mọi camera cùng góp, đổi lại trễ thêm bằng camera chậm nhất.

> **Lưu ý quan trọng:** đo ngày 03/10 cho thấy webcam `front` trả khung **chậm hơn D455 ~86 ms, chậm hơn D435i
> ~54 ms**, nhưng `latency_s` của cả ba camera vẫn đang là `0`. Vì ghép theo thời điểm đến máy, khung camera phụ
> được ghép thực chất là ảnh chụp **sau** ảnh `front` 54–86 ms. Xem [mục 20](#20-hướng-cải-tiến-đề-xuất).

### 8.2 Mô hình camera và hiệu chuẩn

`CameraModel`: `R, t` (thế giới → camera), `K, dist` (webcam: từ `calibrateCamera`; RealSense: intrinsics SDK, chiếu/
khử méo bằng hàm của librealsense). `normalize(px)` → toạ độ chuẩn hoá đã khử méo; `project(X)` → pixel.
Webcam bị chặn nếu chạy độ phân giải khác lúc hiệu chuẩn.

### 8.3 Khoá cùng một người ở mọi camera (`_match_operator`)

Tránh triangulate vai của hai người khác nhau:

1. **Đã có vai 3D gần đây** (≤ `ref_keep_s` 0,5 s): chiếu tâm vai vào camera phụ, chọn người có tâm vai gần điểm
   dự đoán nhất, nhận nếu < `lock_dist` × bề rộng vai quy ra pixel ở khoảng cách đó.
2. **Chưa có**: triangulate vai/khuỷu/hông của từng ứng viên với người camera 0 đang khoá, yêu cầu sai số chiếu
   lại trung vị ≤ 30 px, bề rộng vai 3D trong 0,2–0,6 m và lệch ≤ 35 % so với camera 0, depth camera phụ (nếu có)
   khớp ≤ 25 cm.

Không ai khớp → camera phụ coi như không thấy người lần này.

### 8.4 Hợp nhất một điểm (`fuse_point`) — thuật toán trung tâm

Đầu vào mỗi camera: toạ độ chuẩn hoá `(x, y)`, độ tin cậy MediaPipe, trọng số tin cậy camera; tuỳ chọn điểm depth 3D.

```
1. ≥ 2 camera: DLT không đồng nhất có trọng số
       mỗi camera 2 hàng: w·(x·R[2] − R[0]) X = w·(t[0] − x·t[2]),  tương tự cho y
   → lstsq. Sai số chiếu lại quy đổi về ảnh 640×480 (f = 600 px, 1 px ≈ 0,1°).
2. Sai số lớn nhất > reproj_thresh_px (25):
       còn > 2 camera → bỏ camera tệ nhất, làm lại;
       còn 2 camera   → thử từng "1 camera + depth của chính nó", chọn phương án chiếu khớp nhất với camera kia
                        (độ tin cậy × 0,5); không có depth → độ tin cậy × 0,3.
3. Depth của các camera đã dùng:
       lệch nghiệm 2D < 4 cm     → thêm làm ràng buộc phụ (trọng số 0,3) rồi giải lại;
       GẦN camera hơn nghiệm 2D  → vật che phía trước (ngón che ngón) → bỏ depth, không phạt;
       XA hơn                    → mâu thuẫn (lỗi dọc đường epipolar / depth rơi vào nền) → conflict, × 0,5.
4. Chỉ 1 camera: dùng được nếu có depth, độ tin cậy × mono_depth_conf (thân: 0,5 → thường dưới ngưỡng → khớp giữ).
```

Với điểm thân, độ tin cậy còn nhân `err_conf_factor`: 1 khi sai số ≤ 10 px, giảm tuyến tính tới 0,3 ở 30 px
(18 px → ×0,6 → dưới `min_conf` 0,6 → khớp giữ).

### 8.5 Khung thân

`body_frame` từ vai/hông triangulate; làm mượt SLERP 0,35; nhảy > 45° thì giữ khung cũ, 10 khung liền vẫn nhảy
thì nhận khung mới (người quay người thật). Mất vai → khung thân reset.

### 8.6 Bàn tay 3D (`_fuse_hand`)

1. **Gác cổng** (`_gate_hands`): bàn tay ở camera nào mà cổ tay của nó cách cổ tay Pose (đã triangulate, chiếu vào
   camera đó) > 60 px → bỏ bàn tay ở camera đó.
2. **Trọng số camera** cho bàn tay: `weight` cấu hình × kích thước bàn tay trên ảnh (nhỏ → tối thiểu 0,4) ×
   mức "nhìn thẳng lòng bàn tay" của khung trước (`0,25 + 0,75·|n·hướng nhìn|`).
3. Triangulate **21 điểm** bằng `fuse_point`.
4. **`HandShape`**: học trung vị độ dài 20 đốt xương; đốt nào ngoài 0,6–1,6 lần thì bỏ điểm đầu đốt (đánh dấu thiếu,
   không bịa điểm).
5. **Kẹp**: tỉ số từ 4 điểm 3D, độ tin cậy riêng (`conf.grip` = min độ tin cậy 4 điểm). Thiếu điểm 3D → lấy tỉ số
   từ điểm world MediaPipe của một camera.
6. **`PalmModel`** — hướng lòng bàn tay:
   - 15 khung đầu (thấy rõ lòng bàn tay): dùng khung 3 điểm (`palm_frame_from_depth`) và **học khuôn** 5 điểm lòng
     bàn tay (cổ tay + 4 gốc ngón) trong khung lòng bàn tay (trung vị).
   - Sau đó: **Kabsch có trọng số** giữa khuôn và 5 điểm hiện tại; bỏ dần điểm lệch > 15 mm (giữ ≥ 3 điểm, luôn
     giữ cổ tay); RMS > 12 mm → bỏ khung (`BAD-FIT`).
7. **Kiểm tra giải phẫu**: góc giữa hướng ngón và cẳng tay > 100° (cổ tay người không gập được) → quan sát sai.
8. **Nguồn phụ**: hướng tay từ điểm world MediaPipe từng camera (trọng số 0,5) và từ 5 điểm depth của từng
   RealSense (0,6).
9. **`OrientationFusion`** (máy trạng thái, mục 8.7).
10. Cổ tay chung: trộn cổ tay Pose với cổ tay Hand 70 % nếu hai điểm cách < 8 cm, tỉ lệ trộn đổi tối đa 0,15/khung —
    để J3/J4 (cẳng tay) và J5–J7 (bàn tay) dùng cùng một điểm cổ tay.

### 8.7 Máy trạng thái hướng bàn tay (`OrientationFusion`)

Mặt lòng bàn tay gần phẳng nên luôn có nhập nhằng **lật 180° quanh trục ngón** (nhầm trỏ ↔ út). Thuật toán:

```
 giả thuyết = {base, base·FLIP}
 điểm(c) = tw·(góc(R_trước, c)/90°)² + Σ w_i·(góc(nguồn_phụ_i, c)/90°)²        → chọn nhỏ nhất
 đổi hướng > 100°           → SWITCH? (độ tin cậy 0, cổ tay đứng yên) cho tới khi lặp 3 khung
 mọi nguồn phụ phản đối     → CONFLICT (độ tin cậy 0); 3 lần liền → bỏ hướng cũ
 trộn base + nguồn đồng ý (≤ 70°) bằng SLERP theo trọng số; làm mượt α 0,65 → 0,92 khi xoay nhanh
 vừa thấy lại tay           → ACQUIRE (độ tin cậy 0) tới khi hướng ổn định < 20° đủ 4 khung
 mất tay                    → HOLD ≤ 12 khung (độ tin cậy 0) rồi LOST
```

| Trạng thái | Độ tin cậy | Hệ quả với J5–J7 |
| --- | --- | --- |
| `TRACKING` (≥ 2 nguồn đồng ý hoặc ≥ 3 điểm lòng bàn tay 2 camera xác nhận) | 0,9 | Bám, được phép nhận bước nhảy lớn |
| `DEGRADED` (1 nguồn) | 0,7 | Bám cử động nhỏ, **không** nhận bước nhảy lớn (`jump_confirm_conf` 0,8) |
| `SWITCH?`, `CONFLICT`, `ACQUIRE`, `HOLD` | 0 | Đứng yên |
| `LOST` | — | Đứng yên, cần ACQUIRE lại |

Độ tin cậy `conf.hand` cuối = min(độ tin cậy trạng thái, độ tin cậy điểm lòng bàn tay) nếu nguồn chính là 3D.

### 8.8 Chế độ `front` (không dùng hiện tại)

Vai/khuỷu/cổ tay lấy từ Pose camera 0 (điểm world MediaPipe), camera phụ chỉ chạy Hand → nhẹ hơn gần một nửa,
mượt hơn nhưng độ sâu cánh tay kém. Gác cổng bàn tay camera phụ bằng 5 điểm lòng bàn tay triangulate với camera 0
(≤ 18 px). Đây là cách bản Openarm_Teleop cũ chạy.

---

## 9. Pipeline: độ tin cậy, lọc điểm, retarget, lọc khớp

`ShadowPipeline.step(frame)` cho mỗi tay robot:

### 9.1 Độ tin cậy theo nhóm khớp

| Nhóm | Khớp | Độ tin cậy |
| --- | --- | --- |
| upper | J1, J2 | min(vis vai, vis khuỷu) |
| fore | J3, J4 | min(vis khuỷu, vis cổ tay) |
| hand | J5, J6, J7 | `conf.hand` (mục 8.7) |
| grip | kẹp | `conf.grip` (fusion) hoặc `conf.hand` |

Thêm: J3 × `clip((góc gập khuỷu − 12°)/(30° − 12°), 0, 1)` — tay gần thẳng thì trục J3 trùng trục J5, xoay cánh tay
không xác định → J3 giữ. Khớp có độ tin cậy < `min_conf` 0,6 → bộ lọc giữ giá trị cũ.

### 9.2 Lọc khung xương cánh tay (`ArmShape`, đang bật)

Học trung vị độ dài cánh tay trên / cẳng tay (90 mẫu, cần ≥ 15). Một đoạn lệch > 25 % → **giữ cả J1–J4** khung đó
(một đoạn sai độ dài thường do khuỷu sai, làm sai cả hai hướng). Mất người > 2 s → học lại.

### 9.3 Lọc điểm 3D (`PointKalman`, đang bật)

Kalman vận tốc không đổi cho vai/khuỷu/cổ tay (mỗi toạ độ độc lập): `q = 6 m/s²`, nhiễu đo `r = 1,5 cm / độ tin cậy`.
Mất điểm > 0,3 s → khởi tạo lại. Thay cho EMA `α = 0,8` (mặc định).

### 9.4 Retarget

Xem [mục 10](#10-retarget-kiểu-sew-mimic). Đầu vào: `u = e − s`, `l = w − e` (None nếu nhóm đó kém), `H` (None nếu kém).
Thiếu phần nào thì giữ các khớp tương ứng của nghiệm trước.

### 9.5 Lọc từng khớp (`JointFilter`, 8 phần tử)

Thứ tự cho mỗi phần tử `i`:

```
 NaN hoặc conf < min_conf                → held, bỏ qua
 |x − raw_trước| > jump_deg (35°):
     bắt đầu/đếm lại nếu: chưa đếm, conf < jump_confirm_conf, hoặc giá trị mới chưa ổn định (lệch > jump/2)
     chưa đủ jump_hold_s (0,2 s) hoặc conf thấp → held
     đủ → reset One Euro (nhận bước nhảy)
 One Euro: cutoff = min_cutoff + beta·|dx|
 Vùng chết có trễ: chỉ đổi đầu ra khi lệch > deadband (đầu ra = y − deadband·sign)
```

Tham số mặc định: `min_cutoff` 0,5–0,6 Hz (khớp), 1,5 (kẹp); `beta` 0,02 (khớp) — gần như không nới khi chuyển
động; `deadband` 1,5° (J1, J2, J4), 2,5° (J3), 3° (J5–J7), 0,02 (kẹp); `jump_confirm_conf` 0,7 (J1–J4), 0,8 (J5–J7).

Đầu ra: `targets[side]` (8 số) + `held[side]` (cờ giữ) + `fresh` (khung có ít nhất 1 khớp nhận giá trị mới).

### 9.6 Kẹp (`GripMapper`)

`f = clip((r − pinch)/(open − pinch), 0, 1)`, mặc định pinch 0,25, open 0,9. Tuỳ chọn `levels` (vd `[0, 0.5, 1]`)
với trễ 0,05 và giữ 0,15 s. Không thấy bàn tay → NaN (giữ). Phím **g**: 4 s chụm/xoè hết cỡ, lấy phân vị 5 %/95 %,
chừa lề 8 %.

---

## 10. Retarget kiểu SEW-Mimic

### 10.1 Ý tưởng

Căn **hướng** từng đoạn chi thay vì vị trí đầu kẹp. OpenArm v1.0 có 7 khớp quay, hai trục liên tiếp luôn vuông góc,
và ở `q = 0` trục J3 chạy dọc cánh tay trên, trục J5 chạy dọc cẳng tay. Vì vậy chia bài toán thành chuỗi các bài
toán con dạng đóng:

| Bước | Khớp giải | Điều kiện | Bài toán con |
| --- | --- | --- | --- |
| 1 | J1, J2 | trục J3 (world) ∥ `u` | SP2 |
| 2 | J3, J4 | trục J5 (world) ∥ `l` | SP2; tay gần thẳng (< 12°): giữ J3, giải J4 bằng SP1 |
| 3 | J5, J6 | trục J7 ∥ trục tương ứng của `R_des = H · R_offset` | SP2 |
| 4 | J7 | xoay nốt quanh trục J7 để khung link7 trùng `R_des` | SP1 |

### 10.2 Bài toán con (`geometry.py`)

- **SP1** `(p1, p2, k)`: góc θ quay `p1` quanh `k` gần `p2` nhất — chiếu cả hai lên mặt phẳng ⊥ `k`, `atan2`.
- **SP4** `(p, h, k, d)`: các θ sao cho `hᵀ R(k,θ) p = d` — phương trình `A·[sinθ, cosθ] = b`, tối đa 2 nghiệm;
  vô nghiệm thì trả góc gần nhất.
- **SP2** `(p1, p2, k1, k2)`: các cặp (θ1, θ2) sao cho `R(k1,θ1)p1 ≈ R(k2,θ2)p2`. Vector chung phải thoả hai ràng buộc
  tích vô hướng → mỗi góc là một SP4; ghép cặp theo sai số thật, trả ≤ 2 cặp tốt nhất.

### 10.3 Chọn nghiệm

Mỗi bước có ≤ 2 nghiệm, mỗi góc có các bản tương đương `±2π`. `_pick` chấm điểm
`100 · vi_phạm_giới_hạn + |Δ so với q_prev|` và chọn nhỏ nhất → ưu tiên trong giới hạn, sau đó gần tư thế trước
(chống nhảy nghiệm). `q_prev` là **nghiệm retarget khung trước** (chưa lọc), được seed bằng tư thế robot đo được
lúc engage.

### 10.4 Hướng bàn tay trung tính và `R_offset`

`R_offset = H_trung_tínhᵀ · R_link7(q_ref)` nối hướng bàn tay người ở tư thế trung tính với hướng link7 của robot
khi J5–J7 = 0. `q_ref` = J1–J4 giải từ tư thế tay lúc hiệu chuẩn, J5–J7 = 0 — nên hiệu chuẩn được ở tư thế tay
dang nhẹ, không bắt buộc thả thẳng.

### 10.5 Đánh giá

`RetargetInfo` trả sai số góc giữa hướng chi robot (sau kẹp giới hạn URDF) và hướng chi người: `err_upper`, `err_fore`,
`err_hand` — hiển thị ở chế độ màn hình `full`. Đã thử ngược 500 tư thế: sai lệch < 1e-10° (khi không chạm giới hạn).

### 10.6 Hệ quả thiết kế cần nhớ

- Không phụ thuộc chiều dài tay ✔; nhưng **vị trí kẹp robot không trùng vị trí bàn tay người** → hợp động tác biểu
  diễn, chưa hợp gắp chính xác.
- **J3 nhạy** khi khuỷu gần thẳng (suy biến) — nguồn chính của bước nhảy J3 đo được.
- **J5 nhạy** với hướng bàn tay: trục J5 dọc cẳng tay, lật dấu pháp tuyến lòng bàn tay → J5 đổi ±80–87° (đã quan sát).
- Chạm giới hạn khớp → robot lệch hướng người, không có cơ chế phân bổ lại sai số sang khớp khác.

---

## 11. SafetyGate

Mọi lệnh tới robot (thật hay mô phỏng, kể cả lúc về tư thế nghỉ) đều qua đây. Chạy trong luồng điều khiển.

### 11.1 `step(dt, now)` — thứ tự kiểm tra

```
 chưa engage hoặc chưa có mục tiêu           → giữ lệnh hiện tại ("hold")
 now − t_target > deadman_s (0,4 s)          → giữ ("dead-man")
 ramp = max(0,05, smoothstep((now − t_engage)/engage_blend_s))
 với mỗi tay:
     goal = target (NaN → lệnh hiện tại), kẹp vào soft_limits, kẹp [0,1]
     [mặc định]  cmd += clip(goal − cmd, ± max_vel · ramp · joint_ramp · dt)
     [velocity_tracking] xem 11.3
 chống va chạm hai tay (11.4)
```

- **Ly hợp**: `engage()` / `disengage()`. Khi engage, pipeline được seed bằng tư thế đo được → lệnh bắt đầu đúng
  tư thế hiện tại, không giật.
- **Mục tiêu "không mới"** (`fresh=False`, mọi khớp đang giữ): không làm mới đồng hồ dead-man. Mất mục tiêu
  > 2 × deadman rồi có lại → ramp lại từ đầu.
- **Ramp riêng từng khớp**: khớp đã giữ > `resume_after_s` (0,3 s) rồi chạy lại → hệ số tốc độ khớp đó tăng
  smoothstep trong `resume_blend_s` (1 s). Tránh cổ tay xoay vụt tới hướng mới sau khi mất tay.

### 11.2 Giới hạn

- Giới hạn mềm = giao của `safety.soft_limits_deg` và giới hạn URDF.
- Vận tốc tối đa: mặc định `[45, 45, 60, 60, 90, 90, 90]°/s`; **config đang chạy thật: `[20, 20, 20, 20, 15, 15, 15]°/s`**,
  engage blend 3 s. Kẹp 1,5 đơn vị/s.

### 11.3 Bám theo vận tốc (tắt mặc định, `config/velocity_ff.yaml`)

Ước lượng vận tốc mục tiêu từ hiệu 2 khung camera (bỏ thay đổi < 1°, EMA 0,5), ngoại suy mục tiêu tối đa 70 ms,
`v_des = v_tgt + gain·(goal_ext − q)` (gain 8/s), giới hạn vận tốc và gia tốc (300–1500°/s²), gửi `v` làm `dq` của
lệnh MIT để thành phần `kd` không hãm chuyển động. Mô phỏng: trễ cổ tay 110 → ~0 ms. **Chưa có số đo trên robot thật.**

### 11.4 Chống va chạm hai tay

Mỗi tay 3 capsule (cánh tay trên r = 5 cm, cẳng tay 4,5 cm, bàn tay 5 cm) dựng từ FK của **lệnh**. Bỏ cặp hai
cánh tay trên. Bước mới làm khoảng cách < 3 cm và gần hơn trước → không bỏ cả bước mà thử **từng khớp**, chỉ cho đi
khớp nào không làm khoảng cách giảm dưới ngưỡng. Chỉ kiểm tra tay–tay, chưa có tay–thân, tay–bàn.

---

## 12. Backend robot (CAN-FD)

`OpenArmCANRobot` dùng thư viện `openarm_can` (đối chiếu commit f340d4b), mỗi tay một `_Arm` trên `can0`/`can1`,
chế độ callback STATE.

### 12.1 Vòng đời

```
 connect()  : xả bus → đọc tới khi 2 lần liên tiếp khớp nhau (≤ 1°) → kiểm tra góc trong motor_limits (±5°)
              → in CẢNH BÁO nếu ngoài giới hạn (zero motor sai?)
 [người gõ "yes"]
 enable()   : từ chối nếu ngoài giới hạn hoặc tư thế đã đổi > 3° so với lúc connect
              → enable_all → xả bus → đọc lại qua bộ lọc → kiểm tra phản hồi mới
 send(cmd,dq): kẹp vào motor_limits (theo góc URDF) → URDF→motor, chọn góc tương đương gần góc đo
              → MIT(kp·ramp_gain, kd, q, dq, τ_gravity) cho 7 khớp (+ kẹp nếu bật)
              → recv → lọc số đọc rác → check_fresh (mất phản hồi > 0,1 s → RobotFault)
 park()     : (app.py) tốc độ 15–20°/s về rest_pose qua SafetyGate, bỏ qua ramp; số đọc rác chỉ bỏ, không dừng
 relax()    : kp = 0, giữ kd 1 s (giảm chấn) → disable_all mọi tay (lỗi tay này không bỏ tay kia)
```

`kp` tăng tuyến tính 0 → đủ trong `gain_ramp_s` (1 s) sau enable. Gain chính thức v1.0:
`kp [70, 70, 70, 60, 10, 10, 10]`, `kd [2.75, 2.5, 2.0, 2.0, 0.7, 0.6, 0.5]`; kẹp kp 5, kd 0,1.

### 12.2 Lọc số đọc rác

Ở chế độ STATE, `openarm_can` coi mọi gói DLC ≥ 8 từ ID nhận là gói trạng thái, nên phản hồi ghi tham số Damiao
(byte đầu 0x55) bị giải mã thành góc rác (vd −12,47 rad). Lọc:

- `|q| > 3,7 rad + |offset|` → rác.
- Nhảy > 0,35 rad so với giá trị tốt gần nhất → nghi ngờ; cùng giá trị nhảy lặp **3 lần** liên tiếp (±0,05) → nhận
  (chuyển động thật).
- Một khớp đọc hỏng liên tục > 0,2 s → `RobotFault` (trừ khi đang về tư thế nghỉ).
- Cuối chương trình in số lần đã bỏ.

### 12.3 Quy đổi góc

`motor = sign·URDF + offset`, `URDF = wrap(motor − offset)·sign`. Tay trái v1.0 có J1/J2 lắp lệch 180° (đọc ~178°
là bình thường). Offset phần mềm hiện tại (trung vị 30 mẫu thụ động 03/10, `config/local_both_full.yaml`):

| Tay | J1 | J2 | J3 | J4 | J5 | J6 | J7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| phải | −0,65° | +0,14° | −0,23° | **−5,96°** | +1,93° | −0,19° | +0,25° |
| trái | +0,16° | −0,67° | +1,65° | +0,10° | −1,52° | +3,40° | +4,49° |

J4 tay phải trôi giữa các buổi (−3° ngày 01/10, −6° ngày 03/10) → **đo lại mỗi buổi**.

### 12.4 Robot mô phỏng

`SimRobot`: đo = lệnh vừa gửi (không động lực học). Dùng cho thử pipeline và `--dry-run` (lệnh vào sim, đọc robot
thật để vẽ nét xanh lá).

---

## 13. Hiệu chuẩn

| Loại | Khi nào | Cách | Lưu ở |
| --- | --- | --- | --- |
| Ngoại tham số camera | Mỗi lần dời camera / gập màn hình laptop | `scripts/calibrate_cameras.py`: ChArUco cùng thấy ở camera 0 và camera c → `solvePnP` (trên toạ độ đã khử méo) → `R_c0 = R_c R_0ᵀ`; gộp nhiều lần chụp: chordal mean hướng + trung vị vị trí, bỏ mẫu lệch > max(3°, 2,5×trung vị) | `config/cameras_calib.yaml` |
| Nội tham số webcam | Lần đầu / đổi độ phân giải | `cv2.calibrateCamera` trên ảnh bảng (≥ 8 ảnh), `--redo-intrinsics` | cùng file |
| Độ trễ camera | Khi đổi camera | `scripts/measure_camera_latency.py`: tương quan chéo năng lượng chuyển động (vẫy tay) | `fusion.cameras[i].latency_s` (**chưa áp dụng**) |
| Hướng bàn tay trung tính | Mỗi lần chạy, tự động trước khi engage | Tay buông (cánh tay ≤ 70°, cẳng tay ≤ 80° so với hướng xuống), ≥ 4 ngón xoè, pháp tuyến lòng bàn tay hướng camera (≤ 50°), cử động < 12° suốt 0,6 s và ≥ 5 mẫu → trung bình trực giao hoá → `R_offset`. Phím **c** làm thủ công (chỉ khi đã nhả) | bộ nhớ |
| Kẹp theo người | Tuỳ chọn, phím **g** | Mục 9.6 | in ra terminal → chép vào `grip:` |
| Zero motor / offset | Mỗi buổi | `tools/bringup/read_joints.py` thụ động ở tư thế nghỉ → `urdf_to_motor.offset_deg`. **Không** ghi zero motor trừ khi cả nhóm quyết định | config `local_both_full.yaml` |

Ngưỡng chấp nhận hiệu chuẩn camera (`lenh_duc.md`): sai số chiếu lại nên < 3 px, không dùng nếu > 6 px. Hiện tại
0,52 / 0,64 px.

---

## 14. Máy trạng thái vận hành

```
 khởi động ─► mở camera ─► nạp model ─► connect robot (đọc tư thế, kiểm tra giới hạn)
           ─► [robot thật] gõ "yes" ─► enable (kp ramp 1 s) ─► luồng điều khiển chạy, robot GIỮ tư thế
                                                 │
                ┌────────────────────────────────┘
                ▼
         ┌─────────────┐  tay chưa đạt tư thế calib   ┌──────────────────┐
         │ CHỜ CALIB   │ ───────────────────────────► │ hint trên màn hình│
         │ (cam, vòng) │ ◄─── đạt, giữ 0,6 s ───────  └──────────────────┘
         └─────┬───────┘
               │ cả hai tay đã calib và đang đúng tư thế
               ▼
         ┌─────────────┐  giữ READY liên tục auto_engage_real_s (2 s); mất READY → đếm lại
         │ READY (xanh)│ ───────────────────────────────────────────┐
         └─────┬───────┘                                            │
               │ SPACE (cấm nếu cổ tay được phép cử động mà chưa calib)  │ (chỉ 1 lần mỗi lần chạy)
               ▼                                                    ▼
         ┌──────────────────────────────────────────────────────────────┐
         │ ENGAGE: seed pipeline bằng tư thế đo → ramp 3 s → FOLLOW      │
         │   dead-man / mất tay → khớp giữ; chạy lại → ramp riêng khớp   │
         └─────┬───────────────────────┬────────────────────────────────┘
               │ SPACE                 │ p                         q / Esc / lỗi
               ▼                       ▼                           ▼
         NHẢ (giữ tại chỗ,       về tư thế nghỉ, chạy tiếp    về nghỉ → relax → tắt motor
         không tự engage lại)    (luồng điều khiển mới)       → đóng camera → lưu --record
```

Phím: `SPACE` engage/nhả · `c` hiệu chuẩn tay lại (khi đã nhả) · `g` hiệu chuẩn kẹp · `p` về nghỉ · `q`/`Esc` thoát.

Màn hình: chấm tròn góc phải — cam viền = cần calib, xanh lá = READY / `AUTO SYNC x.xs`, xanh dương = FOLLOW.
Chế độ `compact` (đang dùng) chỉ giữ dòng fps/trạng thái, kẹp, sync/mất khung và chất lượng bàn tay.

---

## 15. Cấu hình

### 15.1 Cơ chế

`config/default.yaml` chứa **mọi** tham số kèm chú thích. Mỗi `--config` ghép đè theo khoá (dict lồng nhau ghép đệ
quy, giá trị khác thay thế), **file sau thắng**. Tham số dòng lệnh `--mode`, `--arms`, `--source` thắng config.

### 15.2 Lệnh chạy thật hiện tại

```bash
.venv/bin/python scripts/shadow.py --source multi --robot openarm \
  --config config/d455_wrist_real.yaml \   # giới hạn đầy đủ tay phải, 20/15°/s, park 15°/s, offset cũ
  --config config/gripper_real.yaml \      # bật kẹp (open −60°, closed 0°)
  --config config/local_both_full.yaml \   # 2 tay J1–J7, offset đo 03/10, auto-engage 2 s, UI compact 10 Hz
  --config config/local_3cam.yaml \        # 3 camera, triangulate, ArmShape + Kalman, pose_interval 2
  --record run_grasp_drop.npz
```

`local_*.yaml` bị `.gitignore` bỏ qua (chứa serial và offset riêng máy) — cần sao lưu thủ công
(`data/2026-10-03_hardware_baseline/` đang giữ một bản).

### 15.3 Danh mục file config

| File | Mục đích |
| --- | --- |
| `default.yaml` | Mọi tham số, giá trị bảo thủ, D455 đơn, robot mô phỏng |
| `first_real.yaml` | Lần chạy thật đầu: chỉ J1–J4, giới hạn hẹp, 20°/s |
| `d455_wrist_real.yaml` | Mở J5–J7 tay phải, toàn dải cơ khí, cổ tay 15°/s |
| `wrist_real_30.yaml`, `wrist_fast.yaml` | Các nấc tăng tốc cổ tay |
| `both_arms_real.yaml`, `both_arms_wrist_real.yaml` | Hai tay (offset 180° tay trái) |
| `gripper_real.yaml` | Bật kẹp |
| `fusion_2cam.yaml`, `fusion_2cam_tri.yaml`, `fusion_3cam.yaml` | Fusion 2 camera (front / triangulate), 3 camera |
| `fusion_real_fast.yaml` | Bộ lọc nhanh hơn (beta 0,5), tốc độ 30–45°/s |
| `velocity_ff.yaml` | Bật bám vận tốc + feedforward |
| `cameras_calib*.yaml` | Kết quả hiệu chuẩn camera |
| `local_3cam.yaml`, `local_both_full.yaml` | Riêng máy hiện tại (không commit) |

---

## 16. Công cụ đo, ghi và phân tích

| Công cụ | Làm gì |
| --- | --- |
| `--record run.npz` | Mỗi khung perception: `t`, `target_*`, `cmd_*`, `conf_*` (8 độ tin cậy), `fus_*` (số camera / sai số px / depth cho vai–khuỷu–cổ tay). Mỗi nhịp điều khiển: `ctl_t`, `ctl_cmd_*`, `ctl_meas_*` |
| `scripts/measure_lag.py` | Từ file record: trễ mục tiêu → lệnh → góc đo theo từng khớp (tương quan chéo) + sai số RMS còn lại sau bù trễ |
| `scripts/find_jumps.py` | Độ phủ tracking, độ tin cậy, sai số chiếu lại, liệt kê bước nhảy > 20°/khung kèm chẩn đoán |
| `scripts/measure_camera_latency.py` | Trễ tương đối giữa các camera |
| `scripts/record_multicam_raw.py` | Ghi RGB (MJPG AVI) + depth thô (uint16 mm) + `timestamps.csv` + nội tham số RealSense (`intrinsics.yaml`) mọi camera, **không bật motor** |
| `scripts/replay_raw.py` | Chạy lại perception + fusion + pipeline trên dữ liệu thô đã ghi (không cần camera/robot), xuất `.npz`; `--compare` so hai lần chạy để kiểm tra hồi quy |
| `scripts/offline_retarget.py`, `replay_npz.py` | Video → quỹ đạo `.npz` → phát lại qua SafetyGate trên robot mô phỏng |
| `scripts/bench_mediapipe.py` | So tốc độ MediaPipe CPU / GPU |
| `scripts/demo_sim.py` | Người giả lập, không cần camera/robot |
| `scripts/check_kinematics.py` | In trục khớp, thử ngược retarget |
| `tools/bringup/` | `setup_can.sh`, `read_joints.py`, `wiggle_j7.py` |

Dữ liệu đã có để phát triển offline:

- `run_grasp_drop.npz`: 136 s chạy thật hai tay (đã xử lý, không có ảnh).
- `data/2026-10-03_raw_bimanual_take2/`: 60 s ảnh + depth thô 3 camera (~30 fps), chuỗi động tác hai tay, xoay cổ
  tay, chụm/xoè, chuyền vật. **Không có timestamp chung** với file chạy robot.

`scripts/replay_raw.py` chạy lại perception trên dữ liệu thô (cùng dữ liệu + cùng config → đầu ra lặp lại y hệt).
Bản ghi `raw_bimanual_take2` không lưu nội tham số RealSense nên đang dùng nội tham số **ước lượng** từ FOV
(`intrinsics_approx.yaml`): dùng được để so A/B, chưa dùng được để đánh giá độ chính xác tuyệt đối. Các bản ghi
mới tự lưu `intrinsics.yaml`.

---

## 17. Hiệu năng đo được và phân tích độ trễ

### 17.1 Số đo ngày 03/10 (`data/2026-10-03_hardware_baseline/`)

| Chỉ số | Giá trị |
| --- | --- |
| Perception | 12,52 fps (~80 ms/khung, 3 camera, Pose 1/2 khung) |
| Vòng điều khiển | 85,99 Hz (đặt 100) |
| Có mục tiêu | ≥ 99 % mọi khớp |
| Độ tin cậy trung vị | phải: J1–J4 0,95–0,98, J5–J7 0,90; trái: J1–J2 0,89, **J3 0,78**, J4 0,92, J5–J7 0,90 |
| Sai số chiếu lại trung vị (vai/khuỷu/cổ tay) | phải 7,3 / 6,6 / 5,6 px; trái 10,0 / 7,3 / 6,0 px |
| Bước nhảy > 20°/khung | phải 13, trái 19 (chủ yếu J3, J5) |
| Trễ camera | front chậm hơn D455 ~86 ms, chậm hơn D435i ~54 ms |

Trễ theo khớp (ms; trong ngoặc: sai số RMS còn lại sau bù trễ):

| Khớp | Phải: mục tiêu→lệnh | lệnh→đo | mục tiêu→đo | Trái: mục tiêu→lệnh | lệnh→đo | mục tiêu→đo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| J1 | 275 | 95 | 370 (10,1°) | 440 | 90 | 520 (4,3°) |
| J2 | 140 | 135 | 270 (5,1°) | 365 | 145 | 555 (5,8°) |
| J3 | 595* | 65 | 595* (9,5°) | 595* | 65 | 595* (19,4°) |
| J4 | 135 | 120 | 245 (12,8°) | 405 | 100 | 540 (11,9°) |
| J5 | 595* | 145 | 595* (20,2°) | 595* | 140 | 595* (16,5°) |
| J6 | 215 | 115 | 330 (7,6°) | 210 | 125 | 330 (8,5°) |
| J7 | 455 | 115 | 580 (4,2°) | 595* | 125 | 595* (6,5°) |

\* chạm trần tìm kiếm 595 ms → trễ thật có thể lớn hơn.

### 17.2 Ngân sách độ trễ (ước tính)

```
 chụp ảnh → đến máy          : webcam front +54…86 ms so với RealSense (camera tham chiếu là camera CHẬM nhất)
 MediaPipe + fusion           : ~80 ms/khung (12,5 fps) + chờ khung kế
 Kalman / One Euro (beta 0,02): bộ lọc gần như không nới khi chuyển động → J1–J4 trễ 130–200 ms (ghi chú trong config)
 SafetyGate                   : giới hạn 20°/s (J1–J4), 15°/s (J5–J7) → cử động lớn bị "cắt" thành dốc
                                (vd J3 biên độ 113° cần ≥ 5,6 s ở 20°/s)
 motor (lệnh → đo)            : 65–145 ms (kp/kd, dq = 0 → kd hãm chuyển động)
```

**Kết luận chính:** phần motor chỉ chiếm 65–145 ms. Phần lớn trễ cảm nhận nằm ở **mục tiêu → lệnh** (giới hạn vận
tốc + bộ lọc) và **trước mục tiêu** (camera + perception). Sai số RMS còn lại lớn ở J3, J4, J5 (10–20°) cho thấy
robot không theo kịp biên độ → giới hạn vận tốc là yếu tố trội trong lần chạy này.

### 17.3 Bước nhảy lớn

- **J5** nhảy tới ±80–87° (chạm giới hạn ±85°) rồi về ~0: dấu hiệu **lật hướng bàn tay** (nhập nhằng 180° quanh
  trục ngón) lọt qua `OrientationFusion`, độ tin cậy vẫn 0,90 (`TRACKING`) → `jump_confirm_conf` 0,8 không chặn.
- **J3** nhảy 20–50°: khi khuỷu gần thẳng hoặc sai số chiếu lại khuỷu/vai tăng (14–25 px trong một số sự kiện).
  Tay trái J3 có độ tin cậy trung vị thấp nhất (0,78).
- Nhiều bước nhảy "giữ/mất 33–104 ms" — tức 1 khung perception: bước nhảy được nhận sau khi giữ đủ `jump_hold_s`.

---

## 18. Các lớp an toàn

Bảo vệ nhiều lớp, từ người tới phần cứng:

| Lớp | Cơ chế | Nơi |
| --- | --- | --- |
| Con người | E-stop trong tay, không ai trong tầm với, checklist `docs/SAFETY.md` | quy trình |
| Khởi động | Gõ `yes`; từ chối enable nếu góc ngoài giới hạn (zero sai) hoặc tư thế đổi > 3° từ lúc đọc đầu; 2 lần đọc khớp nhau | `app.run`, `enable()` |
| Bật motor êm | kp tăng dần 1 s; lệnh đầu = tư thế đo | `send()`, `gate.reset()` |
| Ly hợp | Chưa engage → giữ tư thế; không cho engage nếu cổ tay được phép cử động mà chưa calib tay | `SafetyGate`, `app.run` |
| Ramp | smoothstep 3 s khi engage; ramp riêng khớp vừa chạy lại; ramp lại sau mất mục tiêu | `SafetyGate` |
| Perception | Độ tin cậy từng khớp; xác nhận bước nhảy (giữ 0,2 s + độ tin cậy); ArmShape; HandShape; kiểm tra giải phẫu; khoá người | pipeline, fusion |
| Dead-man | Không có mục tiêu mới > 0,4 s → đứng yên | `SafetyGate` |
| Giới hạn | Giới hạn mềm (SafetyGate) + chốt giới hạn motor (backend), vận tốc, gia tốc (khi bám vận tốc) | |
| Va chạm | Capsule tay–tay, chỉ đi khớp an toàn | `SafetyGate` |
| Phản hồi | Lọc số đọc rác; mất phản hồi > 0,1 s hoặc đọc hỏng > 0,2 s → dừng | backend |
| Kết thúc | Về nghỉ chậm → giảm chấn 1 s → tắt motor, kể cả khi có lỗi | `finally`, `relax()` |

Điểm yếu: mất phản hồi CAN thì tay **hạ xuống từ từ** (giảm chấn), không giữ; không có giám sát mô-men/dòng; không
có watchdog phần cứng phía motor ngoài timeout của Damiao (chưa kiểm chứng trong repo).

---

## 19. Giới hạn đã biết, rủi ro, nợ kỹ thuật

**Thuật toán / chất lượng**

1. Lật hướng bàn tay vẫn lọt qua với độ tin cậy cao → J5 nhảy ±87°.
2. J3 suy biến khi khuỷu gần thẳng; chỉ được xử lý bằng giảm độ tin cậy và giữ.
3. Căn hướng, không căn vị trí → kẹp robot không tới đúng chỗ bàn tay người; gắp chính xác cần điều khiển vị trí
   hoặc lai.
4. Chạm giới hạn khớp: chỉ kẹp, không tìm cấu hình thay thế.
5. Bù trọng lực tắt: với kp 70, tay giơ ngang có thể võng ~8° (ước tính).
6. Chống va chạm chỉ tay–tay.

**Thời gian thực / hiệu năng**

7. `latency_s` chưa áp dụng dù đã đo; camera tham chiếu (webcam) là camera chậm nhất.
8. Perception 12,5 fps trên CPU; Pose `full` × 3 camera là phần nặng nhất.
9. Vòng điều khiển 86 Hz thay vì 100 (tranh GIL khả dĩ), nhịp không đều ảnh hưởng `dt` của SafetyGate.
10. Bộ lọc mặc định (`beta 0,02`) ưu tiên êm hơn nhanh.

**Vận hành**

11. J4 tay phải trôi zero giữa các buổi → phải đo offset mỗi buổi.
12. `local_*.yaml` không nằm trong git → dễ mất cấu hình đang chạy thật.
13. Hiệu chuẩn camera phải làm lại mỗi lần dời camera / gập màn hình laptop.
14. Webcam laptop là camera tham chiếu: chất lượng ảnh thấp nhất nhưng giữ vai trò khung thế giới và khoá người.

**Mã nguồn**

15. ~~`app.run` dài, trộn logic vận hành, UI và ghi log~~ — đã tách (05/10): `runtime/session.py`, `runtime/recorder.py`,
    `viz/hud.py`; phần còn lại trong `run` là nối khối + phím.
16. ~~Hai bộ ổn định hướng song song~~ — đã xoá `HandOrientationTracker` (05/10), chỉ còn `OrientationFusion`.
17. ~~Thiếu công cụ chạy perception offline~~ — đã có `scripts/replay_raw.py` (05/10).
18. Bản dịch bài báo SEW-Mimic chưa có mục IV → cách chia khớp có thể khác tác giả.

---

## 20. Hướng cải tiến đề xuất

Sắp theo tỉ lệ tác động / chi phí. Mỗi mục ghi cách kiểm chứng để quyết định dựa trên số đo.

| # | Cải tiến | Tác động kỳ vọng | Chi phí | Kiểm chứng |
| --- | --- | --- | --- | --- |
| 1 | **Áp dụng `latency_s`** theo số đã đo (script đề xuất −0,086 / −0,054 cho hai RealSense), hoặc dùng camera RealSense làm tham chiếu | Ghép đúng ảnh cùng thời điểm → giảm sai số chiếu lại khi tay di chuyển, giảm bước nhảy | Rất thấp (config + hiệu chuẩn lại nếu đổi tham chiếu) | `find_jumps.py`: sai số px trung vị và số bước nhảy trước/sau |
| 2 | ✅ (05/10, `scripts/replay_raw.py`) **Công cụ perception offline** từ `raw_bimanual_take2` (đọc AVI + depth + timestamps → `MultiSample` → fusion → pipeline → `.npz`) | Tinh chỉnh perception/bộ lọc không cần phần cứng, so sánh A/B lặp lại được | Thấp–trung bình | Chạy cùng dữ liệu với 2 bộ tham số |
| 3 | **Nâng giới hạn vận tốc** theo nấc (20 → 30 → 45°/s) sau khi ổn định; cân nhắc `velocity_ff.yaml` | Giảm trễ mục tiêu → lệnh lớn nhất | Thấp (config), rủi ro an toàn → làm theo nấc | `measure_lag.py`: cột mục tiêu → lệnh và RMS còn lại |
| 4 | **Bộ lọc nhanh hơn** (`beta` 0,2–0,5 cho J1–J4 như `fusion_real_fast.yaml`) | −~70 ms trễ J1–J4, đổi lại rung hơn khi đứng yên | Thấp | `measure_lag.py` + quan sát rung |
| 5 | **Chặn lật J5**: yêu cầu ≥ 2 nguồn đồng ý cho thay đổi hướng lớn; hoặc dùng độ tin cậy thấp hơn khi chỉ có nguồn 3D; hoặc ràng buộc liên tục khớp (từ chối nghiệm làm J5 đổi > X° nếu J6/J7 không đổi tương ứng) | Bỏ phần lớn bước nhảy ±87° | Trung bình | `find_jumps.py` trên dữ liệu offline (mục 2) |
| 6 | **J3 gần suy biến**: thay vì giữ cứng, kéo J3 dần về giá trị "tự nhiên" (vd theo hướng bàn tay hoặc 0) khi khuỷu thẳng | Ít bước nhảy J3 khi gập lại | Trung bình | như trên |
| 7 | **Tăng fps perception**: `pose_landmarker_lite`, giảm độ phân giải webcam đưa vào MediaPipe, `body_source: front` cho camera phụ, hoặc GPU (`bench_mediapipe.py`) | 12,5 → 18–25 fps (ước tính, cần đo) | Thấp–trung bình | fps trên màn hình + sai số px |
| 8 | **Tách vòng điều khiển khỏi GIL** (tiến trình riêng hoặc C++/ROS 2) | Nhịp 100 Hz đều, an toàn hơn khi perception nặng | Cao | `ctl_t` trong record |
| 9 | **Bù trọng lực** (Pinocchio, đã có code) — bật từng khớp, gain thấp | Giảm võng, giảm sai số J2/J4 | Trung bình, cần thử cẩn thận | Sai số lệnh → đo khi giữ tư thế ngang |
| 10 | **Điều khiển lai vị trí–hướng** cho gắp (IK vị trí đầu kẹp có trọng số, giữ dáng tay bằng hướng khuỷu) | Gắp chính xác | Cao | Bài gắp/thả lặp lại |
| 11 | **Va chạm tay–thân / bàn** (capsule thân, mặt phẳng bàn) | An toàn khi nới giới hạn | Trung bình | Test SafetyGate |
| 12 | **Đưa `local_*.yaml` vào quản lý phiên bản** (thư mục `config/site/` có commit, hoặc sao lưu tự động mỗi lần chạy kèm file record) | Không mất cấu hình chạy thật, tái lập được | Rất thấp | — |
| 13 | ✅ (05/10, phần lớn) **Tách `app.run`** thành máy trạng thái vận hành + UI + logger | Dễ kiểm thử, dễ thêm tính năng (auto-engage, chế độ mới) | Trung bình | Test hiện có vẫn đạt |

Gợi ý thứ tự: **1 → 2 → (3, 4) → 5 → 6 → 7**, mỗi bước đo lại bằng `measure_lag.py` / `find_jumps.py` để có số so sánh.

### 20.1 Bài toán chuyền vật từ tay này sang tay kia

**Quan sát khi chạy thật (2 camera depth D435i + D455):** hướng và dáng chuyển động chung thì ổn. Vị trí chưa
chính xác, chưa đủ tin cậy để gắp bằng tay này rồi đưa sang tay kia.

**Phân tích theo kiến trúc hiện tại.** Phần cứng có góp phần, nhưng nguyên nhân lớn nhất nằm ở thiết kế:

1. **Không có vòng nào điều khiển vị trí.** Retarget chỉ căn *hướng* của cánh tay trên, cẳng tay và bàn tay
   (mục 10). Vị trí kẹp robot là kết quả của FK theo chiều dài tay **robot** và khoảng cách vai **robot**. Khi hai
   bàn tay người chạm nhau, hai kẹp robot chỉ chạm nhau nếu tỉ lệ cơ thể người trùng tỉ lệ robot. Bình thường
   chúng không trùng, nên sai số vị trí mang tính **hệ thống** chứ không phải nhiễu, và cải thiện camera không
   loại bỏ được nó.
2. **Sai số hướng cộng dồn theo chuỗi.** Mỗi 1° sai ở J1/J2 làm đầu kẹp lệch khoảng 1,7 cm cho mỗi mét cánh tay
   đòn. Sai số chiếu lại 6–10 px (khoảng 0,6–1°) cùng lệch thời gian camera chưa bù (mục 8.1) cộng lại thành
   vài cm ở đầu kẹp.
3. **Lúc chuyền vật cũng là lúc perception kém nhất.** Hai bàn tay ở gần nhau, che nhau và cầm vật, nên việc gán
   bàn tay theo cổ tay, `HandShape` và Kabsch lòng bàn tay dễ hỏng. Hướng tay rơi về HOLD/ACQUIRE, J5–J7 đứng
   yên, đúng vào lúc cần chính xác nhất. Ngón bị vật che thì tỉ số kẹp cũng sai.
4. **Trễ lớn và giới hạn 20/15°/s.** Robot đi sau người 0,25–0,6 s, nên người phải chờ robot. Khi hai tay đi cùng
   lúc, sai lệch thời gian giữa hai tay thành sai lệch vị trí tương đối.
5. **Không bù trọng lực** (có thể võng khoảng 8° khi tay giơ ngang). **Chống va chạm tay–tay** có thể chặn chính
   động tác đưa hai kẹp lại gần (ngưỡng 3 cm cộng bán kính capsule).

**Hướng xử lý, xếp theo mức độ phù hợp với bài toán:**

| Hướng | Nội dung | Ghi chú |
| --- | --- | --- |
| A. Điều khiển lai vị trí + hướng | Lấy vị trí cổ tay/bàn tay người (khung thân, đã có từ triangulate), co giãn theo tỉ lệ tay robot / tay người, giải IK vị trí đầu kẹp có trọng số và dùng hướng khuỷu làm ràng buộc phụ (giữ dáng). | Sửa đúng nguyên nhân 1. Chi phí cao nhất nhưng là thay đổi cần thiết nếu muốn gắp chính xác |
| B. Căn vị trí **tương đối** giữa hai tay | Khi hai bàn tay người gần nhau (< ngưỡng), chuyển sang chế độ bám vector cổ tay trái → cổ tay phải: tay nhận giữ nguyên, tay đưa chạy IK tới điểm đối diện kẹp kia. | Đúng bản chất bài chuyền vật và ít nhạy với sai số tuyệt đối. Có thể làm sau A, hoặc thay A cho riêng pha chuyền |
| C. Chia sẻ quyền điều khiển (shared autonomy) | Người dẫn tới vùng chuyền; khi hai kẹp cách nhau < vài cm, robot tự căn chỉnh cuối bằng FK của chính nó (đã biết chính xác vị trí hai kẹp), sau đó đóng/mở kẹp theo trình tự. | Rất tin cậy, không phụ thuộc perception ở pha khó nhất |
| D. Cải thiện đầu vào | Áp `latency_s`, nới tốc độ theo nấc, bộ lọc nhanh hơn, chặn lật J5 (mục 20 #1, #3–#5); kẹp theo mức `levels: [0, 1]` khi cầm vật. | Cần làm trong mọi phương án nhưng một mình không đủ |
| E. Nới chống va chạm trong pha chuyền | Giảm `margin_m` / bán kính capsule bàn tay khi đang chuyền, hoặc chỉ kiểm tra cánh tay trên và cẳng tay. | Bắt buộc nếu kẹp phải chạm nhau; cần người giữ E-stop |

**Cách đo để quyết định:** thêm vào `--record` vị trí FK của hai đầu kẹp và vị trí hai bàn tay người (khung thân),
rồi tính sai số khoảng cách kẹp–kẹp so với bàn tay–bàn tay khi hai tay người chạm nhau. Nếu sai số có độ lệch
trung bình lớn (lệch hệ thống) thì nguyên nhân 1 là chính → chọn A/B/C. Nếu trung bình nhỏ nhưng phương sai lớn
thì nguyên nhân là nhiễu perception → ưu tiên D.

---

## 21. Bảng tra cứu tham số

Các núm vặn có ảnh hưởng lớn nhất (giá trị: mặc định → đang chạy thật nếu khác).

| Tham số | Giá trị | Tăng thì | Giảm thì |
| --- | --- | --- | --- |
| `safety.max_vel_deg_s` | 45/60/90 → **20/15** | Bám nhanh hơn, rủi ro cao hơn | Chậm, trễ lớn |
| `safety.engage_blend_s` | 1,5 → **3,0** | Engage êm hơn | Engage nhanh hơn |
| `safety.deadman_s` | 0,4 | Chịu mất khung lâu hơn | Dừng nhạy hơn |
| `filter.beta` (J1–J7) | 0,02 | Ít trễ khi chuyển động, rung hơn | Êm, trễ |
| `filter.min_cutoff` | 0,5–0,6 | Ít trễ, rung hơn | Êm hơn |
| `filter.deadband_deg` | 1,5–3 | Ít rung motor, mất cử động nhỏ | Nhạy cử động nhỏ |
| `filter.jump_deg` / `jump_hold_s` | 35° / 0,2 s | Ít chặn, nhảy dễ lọt | Chặn nhiều, cử động nhanh bị trễ |
| `filter.jump_confirm_conf` | 0,7 / 0,8 | Chặt hơn khi nhận bước nhảy | Lỏng hơn |
| `filter.min_conf` | 0,6 | Giữ khớp nhiều hơn | Đi theo điểm kém |
| `retarget.elbow_straight_deg` / `elbow_j3_full_deg` | 12 / 30 | J3 giữ ở dải rộng hơn | J3 phản ứng sớm, dễ nhảy |
| `fusion.reproj_thresh_px` | 25 | Ít loại camera | Loại camera mạnh tay hơn |
| `fusion.max_skew_s` | 0,04 | Ghép khung lệch nhiều hơn | Bỏ camera phụ nhiều hơn |
| `fusion.cameras[i].latency_s` | **0** (đo được −0,086 / −0,054) | — | — |
| `fusion.orientation_fusion.switch_deg` / `switch_frames` | 100 / 3 | Khó đổi hướng hơn | Dễ lật |
| `models.pose_interval` | 1 → **2** | Nhẹ CPU, thân cập nhật thưa | Nặng, mượt |
| `calibration.hand_auto.auto_engage_real_s` | null → **2,0** | Phải giữ READY lâu hơn | Engage nhanh |
| `robot.kp` / `kd` | chính thức v1.0 | Cứng hơn | Mềm, võng |

---

## 22. Kiểm thử

- `pytest`: 21 file, 158 hàm test (không cần camera/robot). Bao phủ: động học, retarget (thử ngược), SP1/SP2, bộ lọc,
  xác nhận bước nhảy, SafetyGate (ramp, dead-man, va chạm, bám vận tốc, resume), backend CAN với `openarm_can` giả
  lập (lọc số rác, enable), fusion (triangulate, khoá người, sync), làm sạch bàn tay, hướng tay, kẹp, khoá người
  1 camera, layout UI, đo trễ.
- Chưa có: test tích hợp với dữ liệu camera thật ghi sẵn; test cho máy trạng thái vận hành trong `app.run`
  (auto-engage, phím).
- Quy trình trước khi chạy thật: `pytest` → `demo_sim.py` → `shadow.py` (sim, camera thật) → `--dry-run` → robot thật
  với config bảo thủ (`docs/SAFETY.md`).

---

*Tài liệu liên quan:* `README.md` (cài đặt, lệnh), `docs/DESIGN.md` (khung toạ độ, retarget), `docs/FUSION.md`
(fusion, đọc màn hình, xử lý sự cố), `docs/SAFETY.md` (checklist), `lenh_duc.md` (lệnh chuẩn cho bộ phần cứng hiện tại).

---

## 23. Phụ lục giải thích dành cho người mới

Phần này bổ sung theo mục góp ý **“5. Người mới còn cần được giải thích thêm những gì?”**. Số 5 đó là số mục
trong bản nhận xét, không phải mục 5 “Hành trình của một khung hình” của tài liệu. Nội dung các mục 1–22 được
giữ nguyên; phụ lục chỉ bổ sung nền tảng để đọc dữ liệu, hình hiển thị, thuật toán và các vòng phản hồi.

Có thể đọc phụ lục trước, rồi quay lại mục 3–11 để hiểu chi tiết triển khai. Ví dụ số dưới đây là **ví dụ minh
hoạ**, không phải số đo mới, thông số hiệu chuẩn thật hay lệnh được phép gửi tới robot.

### 23.1. Từ điển dữ liệu: mỗi tầng đang truyền cái gì?

Một ảnh không biến trực tiếp thành lệnh motor. Chương trình lần lượt tạo ra **quan sát người**, **nghiệm khớp
robot**, **mục tiêu đã lọc**, **lệnh được phép thực thi** và **số đo phản hồi**. Các dữ liệu này có ý nghĩa khác nhau.

#### 23.1.1. Ảnh, điểm và hướng

| Tên trong code | Ý nghĩa, dạng và đơn vị | Cách hiểu đúng |
| --- | --- | --- |
| `CameraSample.bgr` | Ảnh màu, mảng cao × rộng × 3, thứ tự kênh BGR | Là dữ liệu ảnh, chưa có vai/khuỷu/cổ tay |
| `CameraSample.depth_m` | Bản đồ depth, đơn vị mét, hoặc `None` | Với nguồn RealSense đang dùng, depth được align sang ảnh màu; không phải tọa độ 3D hoàn chỉnh của từng pixel |
| `intrinsics`, `K`, `dist` | Nội tham số và mô hình méo của camera | Dùng để nối pixel với tia nhìn; không cho biết camera nằm ở đâu so với camera khác |
| `CameraModel.R`, `CameraModel.t` | Phép quay 3×3 và tịnh tiến 3 phần tử, với `X_cam = R @ X_world + t` | `t` ở đây là vector mét, khác `Frame.t` là thời gian |
| `MultiSample` | Nhóm ảnh các camera, thời gian tham chiếu, độ lệch và cờ `stale` | Một nhóm ảnh ghép được không đồng nghĩa tất cả camera đã phơi sáng cùng lúc |
| `pose_2d`, `hands_2d` | Landmark ảnh; x/y của MediaPipe trong các trường này được chuẩn hoá theo kích thước ảnh | Muốn thành pixel phải nhân chiều rộng/cao; không dùng trực tiếp như mét |
| `W` trong fusion | Mảng điểm người 3D, đơn vị mét trong khung camera tham chiếu | Có thể chứa `NaN` ở landmark chưa khôi phục được; không phải tọa độ đế robot |
| `Frame.body_origin` | Gốc khung thân trong khung tham chiếu, đơn vị mét | Trong đường triangulate, đặt tại giữa hai vai |
| `Frame.body_R` | Ba cột là các trục thân biểu diễn trong khung tham chiếu | Dùng để đổi điểm/hướng sang khung thân, không phải phép hiệu chuẩn camera–robot |
| `ArmObs.s`, `.e`, `.w` | Vai, khuỷu, cổ tay; mỗi điểm gồm 3 số mét trong khung thân | Đây là ba điểm của **người**; `w` không phải TCP robot |
| `u`, `l` | Vector vai→khuỷu và khuỷu→cổ tay | Trong `pipeline.step` lấy bằng hiệu hai điểm; retarget chuẩn hoá để dùng hướng, không dùng chiều dài để đặt TCP |
| `ArmObs.H` | Ma trận quay 3×3: các trục bàn tay trong khung thân | Chứa **hướng**, không chứa vị trí bàn tay; không phải ba góc Euler |
| `ArmObs.grip` | Tỉ số khoảng cách ngón cái–trỏ / chiều dài bàn tay | Là tỉ số quan sát chưa ánh xạ, có thể lớn hơn 1; **chưa phải** lệnh độ mở kẹp 0..1 |
| `ArmObs.conf` | Điểm tin cậy các nhóm `upper`, `fore`, `hand`; fusion bổ sung `grip` riêng | Là chỉ báo để gate/lọc, không phải xác suất chính xác đã được hiệu chuẩn |

Hai loại “tọa độ chuẩn hoá” rất dễ bị nhầm:

- Landmark chuẩn hoá theo ảnh: ví dụ `(0,5; 0,5)` nghĩa là giữa ảnh, trước khi nhân kích thước ảnh.
- Điểm sau `CameraModel.normalize()`: `(X/Z, Y/Z)` đã xét nội tham số/khử méo; không bị giới hạn trong 0..1.
  Đây mới là dạng dùng để lập các phương trình triangulation.

MediaPipe còn có đầu ra mang tên `world landmarks`. Tên `world` của model **không có nghĩa** nó đã nằm trong
khung thế giới fusion hay khung robot. Pose/Hand có gốc và cách ước lượng riêng; code phải biến đổi hoặc hợp
nhất phù hợp. Không cộng vị trí `world` của hai camera như thể đã có cùng gốc tọa độ.

#### 23.1.2. Nghiệm, mục tiêu, lệnh và số đo

| Tên | Ai tạo, ai dùng? | Không được nhầm với |
| --- | --- | --- |
| `q` trả từ `ArmRetargeter.solve()` | Nghiệm 7 góc theo URDF, đã clamp giới hạn động học; đưa vào bộ lọc | Tư thế robot đã thực sự đạt |
| `q_prev` | Nghiệm retarget trước đó để chọn nhánh gần nhất; được seed từ số đo ở các thời điểm tương ứng | Feedback cập nhật liên tục mỗi lần retarget |
| `GripMapper` trả `g` | Tỉ số ngón được ánh xạ thành độ mở chuẩn hoá 0..1, hoặc `NaN` khi không có quan sát hợp lệ | Góc motor kẹp, độ mở mm hay lực giữ vật |
| `targets[side]` | Đầu ra `JointFilter`, làm đích cho `SafetyGate` | Lệnh đã qua giới hạn tốc độ/va chạm của SafetyGate |
| `cmd[side]` | Lệnh từ SafetyGate cho nhịp control hiện tại | Số đo robot; backend còn có chốt giới hạn motor và đổi hệ góc |
| `q_meas`, `ctl_meas_*` | Số đo motor được backend kiểm tra/lọc rồi quy đổi về hệ URDF và độ mở kẹp | Đo độc lập vị trí TCP bằng camera/thước |
| `dq` gửi backend, nếu có | Vận tốc mong muốn của 7 khớp, rad/s | Đạo hàm đã đo của robot; khi không cấp `dq`, backend gửi giá trị vận tốc mong muốn bằng 0 |
| `held[side]` | 8 cờ của bộ lọc: phần tử nào bị giữ do dữ liệu không đạt điều kiện | Lệnh phanh tức thì hay chứng nhận khớp đã ngừng chuyển động |
| `pipeline.fresh` | Cờ tổng hợp từ các cờ giữ của J1–J7 ở các tay | Chứng nhận mọi tay, mọi nguồn và kẹp đều có quan sát mới |

Đối với `targets`, `cmd` và số đo backend, một tay thường có dạng:

```text
[q1, q2, q3, q4, q5, q6, q7, g]
 └──────── rad theo URDF ────────┘  └ độ mở chuẩn hoá: 0 đóng, 1 mở

chỉ số Python: 0..6 là J1..J7; chỉ số 7 là kẹp
```

Ví dụ `q4 = 1,5708 rad` gần bằng 90°. Nhưng `g = 0,5` là nửa **khoảng điều khiển đã cấu hình** của kẹp, không
phải 0,5 rad hay 50% lực. Không gọi `rad2deg` cho cả 8 phần tử rồi đọc phần tử cuối như góc kẹp.

Ví dụ phân biệt các tầng ở một khớp, giả sử đã hết ramp và không có chặn khác:

```text
target = 30°; cmd hiện tại = 10°; số đo gần nhất = 9°
max_vel = 20°/s; dt = 0,01 s
→ bước command tối đa = 0,2°; command tiếp theo hướng tới 10,2°, không nhảy ngay lên 30°.
```

Nếu bộ lọc giữ target cũ ở 30°, command vẫn có thể tiếp tục tiến tới 30° khi SafetyGate còn cho phép. Đây là
lý do phải phân biệt **giữ mục tiêu**, **giữ lệnh** và **robot đứng yên ngoài thực tế**. `NaN` nghĩa là không có
giá trị hợp lệ cho tầng đó, không phải góc zero; sau khi đã có giá trị, bộ lọc thường trả lại số hữu hạn cũ.

#### 23.1.3. Thời gian và dữ liệu cũ

`Frame.t`/`MultiSample.t` trong đường nhiều camera xuất phát từ thời gian host đọc khung tham chiếu, có trừ
`latency_s`. Nó không tự trở thành thời điểm phơi sáng chính xác. `CameraSample.timestamp_s` là metadata nguồn
nếu có, không phải mọi nguồn đều có hoặc cùng một đồng hồ.

Trong log, `t` gắn với frame perception; `ctl_t` gắn với nhịp control. Hai chuỗi có nhịp và số mẫu khác nhau.
Không lấy phần tử thứ 100 của chúng ghép thành một thời điểm nếu chưa căn timestamp. `robot.read()` trả trạng
thái đang lưu ở backend; gọi lại hàm không đồng nghĩa vừa nhận được một gói CAN mới.

Các từ hay gặp:

- **Observed:** có quan sát từ ảnh ở lần đo đó.
- **Reused/held:** dùng lại kết quả trước; giá trị có thể vẫn đẹp và hữu hạn nhưng tuổi dữ liệu đã tăng.
- **Predicted:** kết quả suy từ mô hình chuyển động; không được coi là phép đo mới.
- **Stale:** quá cũ đối với điều kiện sử dụng; có timestamp mới ở tầng sau cũng không làm trẻ lại ảnh gốc.

Đây là cách phân biệt về mặt khái niệm; không phải hiện tại mọi cấu trúc dữ liệu/log đều đã có đủ bốn nhãn
này. Riêng `fresh` hiện được OR giữa các tay và không tính riêng kẹp: còn một khớp arm không bị `held` là cờ
có thể đúng. `held=False` cũng không bảo đảm góc đầu ra đã đổi, ví dụ deadband có thể vẫn giữ đầu ra cũ.

Nguồn tra cứu: [sources.py](openarm_shadow/cameras/sources.py), [perception.py](openarm_shadow/perception/landmarker.py),
[multiview.py](openarm_shadow/fusion/multiview.py), [pipeline.py](openarm_shadow/retarget/pipeline.py),
[app.py](openarm_shadow/runtime/app.py), [backend CAN](openarm_shadow/robot/openarm_can_robot.py).

### 23.2. Một ví dụ đi qua các hệ tọa độ

Điều quan trọng không chỉ là “điểm có ba số”, mà là **ba số đo từ gốc nào, theo trục nào và đơn vị gì**.
Phần này dùng vector cột cho công thức. Code có lúc lưu nhiều điểm theo hàng của mảng NumPy nên phép nhân
nhìn như bị đảo/transposed; hai cách biểu diễn phải cho cùng kết quả hình học.

#### 23.2.1. Từ pixel tới một tia nhìn, rồi mới tới điểm 3D

Với camera giả định không méo, `fx = fy = 600 px`, `cx = 320 px`, `cy = 240 px`, pixel `(380, 300)` cho:

```text
x_n = (380 - 320) / 600 = 0,1
y_n = (300 - 240) / 600 = 0,1
tia nhìn có hướng tỉ lệ với [0,1; 0,1; 1]
```

Pixel đó có thể là điểm gần hoặc xa nằm trên cùng tia; **một pixel không tự cho biết độ sâu**. Nếu có depth
`Z = 2 m` hợp lệ theo trục z camera, phép nhấc lên 3D trong ví dụ này cho `X_cam = (0,2; 0,2; 2) m`. Z không
phải khoảng cách Euclid từ tâm camera tới điểm. Nếu dùng nhiều camera, triangulation tìm điểm phù hợp với các
tia quan sát; ngoại chuẩn cho biết các tia bắt đầu ở đâu và quay hướng nào so với nhau.

#### 23.2.2. Đưa các camera về cùng một khung

Với camera phụ c, config/model dùng quy ước:

```text
X_c = R_c @ X_world + t_c
X_world = R_c.T @ (X_c - t_c)
```

`world` ở đây là camera `front` được chọn làm tham chiếu. Phép đổi này nối **camera với camera**, không nối
camera với đế robot. ChArUco được dùng để tìm quan hệ giữa camera; khung bảng không tự trở thành khung điều
khiển robot cố định theo mặt bàn.

#### 23.2.3. Từ khung camera tham chiếu sang khung thân người

Giả sử người đứng thẳng nhìn vào camera; gốc giữa vai trong khung tham chiếu là `O = (0; 0; 2) m`. Trong ví dụ
này: phía trước người là hướng về camera, phía trái người là phía phải ảnh chưa lật gương, phía trên người
ngược trục y của camera. Khi đó:

```text
R_body = [ 0   1   0 ]     các cột lần lượt là x_thân, y_thân, z_thân
         [ 0   0  -1 ]     biểu diễn trong khung camera tham chiếu
         [-1   0   0 ]

X_body = R_body.T @ (X_world - O)
H_body = R_body.T @ R_hand_world
```

Điểm cần trừ gốc O; hướng chỉ cần quay, không trừ gốc. Áp dụng cho ba điểm tay phải giả định:

| Điểm | Trong camera tham chiếu, mét | Trong khung thân, mét |
| --- | --- | --- |
| Vai `s` | `(−0,20; 0; 2,00)` | `(0; −0,20; 0)` |
| Khuỷu `e` | `(−0,20; 0,30; 2,00)` | `(0; −0,20; −0,30)` |
| Cổ tay `w` | `(−0,20; 0,55; 1,85)` | `(0,15; −0,20; −0,55)` |

Diễn giải: tay phải nằm phía y âm; khuỷu thấp hơn vai 30 cm; cổ tay nằm trước gốc thân 15 cm. Từ đó:

```text
u = e - s = (0; 0; -0,30)       → hướng đơn vị (0; 0; -1)
l = w - e = (0,15; 0; -0,25)   → hướng đơn vị xấp xỉ (0,514; 0; -0,857)
```

Retarget nhận các **hướng** này và `H_body`, tìm góc khớp để các trục chi robot căn theo chúng. Robot có chiều
dài link khác nên TCP của nó không bắt buộc nằm tại `(0,15; −0,20; −0,55)`. Phần đặt vị trí đế, chiều dài link
và offset TCP thuộc mô hình robot, không được thay bằng khoảng cách vai hay chiều dài tay người.

#### 23.2.4. Từ góc URDF tới motor, và các loại “zero” khác nhau

Sau retarget, lọc và SafetyGate, backend đổi góc theo `q_motor = sign × q_URDF + offset` rồi chọn biểu diễn góc
tương đương gần số đo. Ví dụ giả định `sign = −1`, offset 5°, lệnh URDF 20° tương ứng motor −15° trước xử lý
chu kỳ. Đây là minh hoạ quy ước, **không phải sign/offset để chép vào robot hiện tại**.

Cần tách bốn việc:

1. **Calib camera:** để các camera nói về cùng điểm 3D.
2. **Calib hướng bàn tay:** xác định hướng người trung tính tương ứng hướng link robot nào qua `R_offset`.
3. **Offset góc motor:** nối số encoder với góc của mô hình; bù phần mềm không phải ghi lại zero firmware.
4. **Seed nghiệm:** dùng một tư thế tham chiếu để chọn nghiệm gần nó; riêng `pipeline.seed()` chỉ cập nhật
   `q_prev`, không tự reset toàn bộ bộ lọc hay mọi trạng thái command.

Do đó, calib bảng tốt không chứng minh zero robot đúng; seed từ robot không tạo ra hiệu chuẩn camera–robot;
và một tư thế nghỉ thẳng không chứng minh TCP đúng ở mọi tư thế. Nguồn: [multiview.py](openarm_shadow/fusion/multiview.py),
[retarget.py](openarm_shadow/retarget/sew.py), [kinematics.py](openarm_shadow/core/kinematics.py).

### 23.3. Đọc màn hình: quan sát người, hình que robot và TCP thật

Màn hình có hai nhóm thông tin khác bản chất:

- **Nét trên ảnh người:** landmark/khung xương do perception ước lượng. Bám đúng ảnh là kiểm tra hữu ích nhưng
  chưa chứng minh chiều sâu hoặc hướng lòng bàn tay 3D đã đúng.
- **Hình que robot:** hình tính từ mô hình và góc khớp. Không phải camera đang nhìn robot và vẽ lại vị trí đo
  được của từng link ngoài đời.

Trong `viz.draw_robot()`:

| Nét trong ô robot | Dữ liệu gốc | Ý nghĩa |
| --- | --- | --- |
| Xám mảnh | `q_target` | Tư thế đích sau bộ lọc pipeline |
| Đậm màu: tay phải xanh dương, tay trái cam | `q` truyền cho hàm vẽ, ở app là `cmd` | Tư thế từ lệnh SafetyGate hiện tại |
| Xanh lá, khi có dữ liệu robot thật | `q_meas` | Tư thế tính từ feedback góc, không phải vị trí TCP đo ngoài |
| Mũi tên x đỏ, y xanh lá, z xanh dương; nhãn U/F/H | Các khung từ **command** | Hướng khung cánh tay trên, cẳng tay và đầu tay; không phải ba bộ trục target/cmd/feedback riêng |

Màu xanh lá trên ảnh người hoặc trên mũi tên trục không mang cùng ý nghĩa với nét xanh lá feedback trong ô
robot. Khi các nét chồng nhau, một nét có thể che nét khác; phải xem dữ liệu số nếu cần phân biệt sai lệch nhỏ.

Đặc biệt, viewer gọi `display_keypoints()`: dựng các đoạn thẳng lý tưởng theo trục J3, J5 và hướng đoạn đầu
tay. Cách vẽ này giúp xem dáng/hướng chi mà không bị các offset gá motor làm hình thành đường zig-zag. Nó
**không dùng nguyên mọi tâm khớp cơ khí làm đường xương**.

Các hàm động học cũng không trả cùng một “đầu tay”:

| Hàm/điểm | Dùng để hiểu điều gì? |
| --- | --- |
| `joint_positions(q)[7]` | Điểm TCP được định nghĩa trong model; chỉ số mảng bắt đầu từ 0 |
| `joint_positions(q)[8]` | Điểm đầu ngón theo offset trong model, khác TCP |
| `keypoints(q)["tool"]` | Hiện lấy điểm đầu ngón `[8]`, dùng cùng các điểm vai/khuỷu/cổ tay cho hình học capsule |
| `display_keypoints(q)["tool"]` | Điểm cuối của hình que lý tưởng để hiển thị; không dùng thay TCP đo/điều khiển |

Vì vậy, “đường xanh đã thẳng” chỉ chứng tỏ góc đo sau quy đổi cho hình chi lý tưởng đang thẳng theo model.
Muốn biết đầu kẹp ngoài đời có tới đúng vị trí/hướng hay không còn cần hình học/TCP đúng, offset đúng và phép
đo ngoài đủ tin cậy. Encoder không tự thấy độ võng, độ rơ, gá kẹp lệch hoặc vật trượt trong kẹp.

`SimRobot` còn đơn giản hơn: số đo được gán bằng lệnh vừa gửi, không mô phỏng quán tính, tiếp xúc, tải hay
ma sát. Đường sim bám đẹp giúp kiểm tra logic, không phải nghiệm thu độ chính xác robot thật. Ở `--dry-run`,
command đi vào sim còn backend thật chỉ được đọc để hiển thị; nét xanh thật không có nghĩa robot đang nhận
lệnh chuyển động của sim.

Nguồn: [viz.py](openarm_shadow/viz/draw.py), [kinematics.py](openarm_shadow/core/kinematics.py),
[SimRobot](openarm_shadow/robot/sim.py), [app.py](openarm_shadow/runtime/app.py).

### 23.4. Hệ thống có vòng phản hồi nào, và chưa có vòng nào?

“Có feedback” chỉ có nghĩa khi nói rõ **đại lượng được đo**, **đại lượng đích** và **nơi dùng sai số để sửa lệnh**.
Hệ hiện tại không hoàn toàn hở vòng, nhưng cũng chưa khép kín ở cấp gắp/chuyển vật.

| Cấp | Có gì hiện tại? | Giới hạn cần hiểu |
| --- | --- | --- |
| Bắt chước người | Camera đo dáng người; retarget sinh góc đích | Camera đang theo người, không tự đo sai số TCP/vật của robot để sửa đích |
| Sinh command trong Python | SafetyGate đưa command tiến tới target theo giới hạn/ramp | Đường mặc định không phải servo Cartesian lấy sai số vị trí TCP thật làm đầu vào |
| Servo motor | Backend gửi MIT với vị trí, vận tốc mong muốn, `kp`, `kd`, mô-men bù; motor dùng feedback nội bộ để bám | Motor bám góc không tự đảm bảo đúng vị trí vật hay đúng lực giữ; hành vi lỗi cần kiểm chứng phần cứng |
| Giám sát trên máy tính | Backend nhận feedback, kiểm tra số đọc/độ mới; số đo dùng lúc khởi tạo/seed, hiển thị và log | Trong đường mặc định, không có bộ điều phối gắp tự dùng sai số TCP đo ngoài để căn hai kẹp |
| Người vận hành | Người quan sát robot/vật rồi sửa động tác | Là vòng phản hồi do con người khép; có độ trễ và không thay E-stop hay giới hạn phần cứng |
| Tác vụ chuyển vật | Chưa có xác nhận tự động đã gắp chắc, biết pose vật và cho phép tay cho nhả | Hai tay người gần nhau hoặc lệnh kẹp đóng không phải bằng chứng robot đã nhận được vật |

Có thể đọc luồng theo sơ đồ nhỏ sau:

```text
người → ảnh → quan sát → retarget/lọc → target → SafetyGate → command → motor → robot
 ↑                                                                        │
 └────────── người quan sát rồi chỉnh động tác ─────────────────────────────┘

feedback encoder → servo bên trong motor
feedback gửi về PC → kiểm tra / khởi tạo–seed / hiển thị / log
chưa có đường tự động: đo TCP/vật ngoài thực tế → sai số Cartesian → sửa target để nhận vật
```

Ở mức khái niệm, thành phần phản hồi của lệnh MIT hướng tới hành vi dạng
`mô-men ≈ kp × sai số góc + kd × sai số vận tốc + mô-men bù`. Đây là cách giải thích vai trò các trường lệnh,
không phải mô hình đã đo đầy đủ firmware/motor. `kp` và `kd` có vai trò khác nhau; tăng chúng không tương đương
tăng độ chính xác camera hoặc sửa quan hệ hình học người–robot.

Vì vậy có thể đồng thời xảy ra: encoder bám command tốt, hình xanh gần hình command, nhưng kẹp vẫn lệch vật.
Khi ấy cần phân biệt sai số **quan sát/mapping**, **target–command**, **command–feedback** và **FK–TCP thật**;
không thể dùng một đồ thị góc khớp để kết luận tất cả các tầng đều chính xác.

Nguồn: [Controller và app](openarm_shadow/runtime/app.py), [SafetyGate](openarm_shadow/safety/gate.py),
[backend CAN](openarm_shadow/robot/openarm_can_robot.py). Phần này giải thích hệ đang có, không có nghĩa các
vòng điều khiển/tác vụ được đề xuất trong tài liệu khác đã triển khai.

### 23.5. Hiểu thuật toán bằng cùng một cách đọc

Với mỗi thuật toán, nên trả lời sáu câu: **giải quyết việc gì → đầu vào → đầu ra → làm bằng cách nào → cần
giả định gì → thất bại thì biểu hiện ở đâu**. Như vậy người mới có thể nối các khối thay vì chỉ nhớ tên DLT,
Kabsch hay One Euro. Dưới đây là cách đọc một số khối chính trong đường nhiều camera hiện tại.

#### 23.5.1. Triangulation: nhiều tia nhìn thành một điểm

- **Việc cần giải:** xác định vị trí 3D của cùng một landmark nhìn từ nhiều camera.
- **Đầu vào:** điểm ảnh đã chuẩn hoá/khử méo, nội–ngoại tham số, trọng số và depth nếu có.
- **Đầu ra:** điểm 3D trong khung tham chiếu, confidence và chẩn đoán camera/sai số/depth.
- **Cách làm:** mỗi camera tạo ràng buộc cho điểm nằm trên tia nhìn. Do nhiễu, các tia thường không giao nhau
  đúng một điểm; bình phương tối thiểu có trọng số tìm điểm phù hợp nhất. `fuse_point` còn thử loại nguồn
  không nhất quán và dùng depth theo điều kiện trong mục 8.4.
- **Giả định:** các camera nhìn cùng người/cùng landmark, hình học calib đúng và thời điểm đủ gần; góc giữa
  các tia phải cung cấp thông tin độ sâu hữu ích.
- **Khi sai:** điểm 3D nhảy hoặc confidence giảm; xem sai số chiếu lại, camera bị bỏ và độ lệch thời gian.
  Residual ảnh nhỏ vẫn có thể đi cùng sai số độ sâu lớn hoặc ghép nhầm có hình học tình cờ phù hợp.

#### 23.5.2. ArmShape/HandShape: kiểm tra tính hợp lý, không “vẽ lại cho đẹp”

- **Việc cần giải:** phát hiện landmark làm xương dài/ngắn bất thường.
- **Đầu vào:** điểm 3D và lịch sử độ dài đã học từ các mẫu được chấp nhận.
- **Đầu ra:** cờ đoạn hợp lệ ở ArmShape; các điểm bị đánh dấu thiếu ở HandShape.
- **Cách làm:** lấy độ dài tham chiếu bằng trung vị, so độ dài hiện tại với khoảng cho phép. ArmShape có thể
  làm pipeline giữ cả nhóm arm; HandShape bỏ điểm nghi ngờ để tầng hướng không dùng nó như quan sát tốt.
- **Giả định:** chiều dài cơ thể ổn định và mẫu học ban đầu đủ đúng.
- **Khi sai:** tracking bị loại nhiều dù ảnh vẫn có landmark; xem độ dài, tỉ lệ so tham chiếu và số điểm bị bỏ.
  Đây là bộ kiểm tra hình học, không phải solver tự khôi phục mọi ngón bị che.

#### 23.5.3. Kabsch: từ các gốc ngón suy ra hướng lòng bàn tay

- **Việc cần giải:** ước lượng hướng bàn tay từ phần lòng bàn tay tương đối cứng.
- **Đầu vào:** khuôn 5 điểm đã học, các điểm 3D hiện tại còn hợp lệ và trọng số.
- **Đầu ra:** ma trận hướng, tâm ước lượng, RMS fit và mode của PalmModel; thiếu dữ liệu thì không có hướng mới.
- **Cách làm:** tìm phép quay và tịnh tiến đặt khuôn trùng các điểm quan sát tốt nhất; dùng SVD cho phần quay,
  có kiểm tra để không trả phép phản xạ. Code thử bỏ điểm lệch lớn và từ chối fit quá kém.
- **Giả định:** landmark được gán đúng ngón/tay, phần còn thấy đủ xác định hướng và khuôn học phù hợp.
- **Khi sai:** `MISSING`/`BAD-FIT`, ít điểm hoặc fit không ổn; nguồn sai nhãn có thể vẫn cho fit nhỏ. Lòng bàn
  tay gần phẳng không đồng nghĩa mọi khung đều mơ hồ: cần phân biệt hình học thật với sai gán/thiếu quan sát.

#### 23.5.4. OrientationFusion và SLERP: chọn hướng đáng tin rồi làm mượt

- **Việc cần giải:** kết hợp các ước lượng hướng và hạn chế flip/reacquire sai.
- **Đầu vào:** hướng 3D chính nếu có, hướng phụ RGB/depth, trạng thái hướng trước và các ngưỡng xác nhận.
- **Đầu ra:** hướng hợp nhất, trạng thái như TRACKING/HOLD/ACQUIRE, confidence và nguồn đã dùng.
- **Cách làm:** so các giả thuyết với lịch sử/nguồn phụ; khi phù hợp thì nội suy các phép quay bằng SLERP.
  SLERP đi trên không gian hướng quay, tránh cách lấy trung bình góc Euler làm sai khi đi qua ±180°.
- **Giả định:** vẫn còn bằng chứng đủ tin để chọn hướng; các nguồn đồng ý không mặc nhiên là độc lập.
- **Khi sai:** hold/acquire kéo dài hoặc hướng sai lọt qua; xem state, nguồn và hướng trước/sau hợp nhất.
  Chống flip không có khả năng nhìn xuyên vật che và không chứng nhận mọi hướng TRACKING là đúng.

#### 23.5.5. PointKalman: phối hợp mô hình chuyển động và điểm đo

- **Việc cần giải:** giảm rung điểm vai/khuỷu/cổ tay trước khi lấy hướng đoạn tay.
- **Đầu vào:** điểm đo, thời gian và confidence; trạng thái trước gồm vị trí/vận tốc ước lượng.
- **Đầu ra:** vị trí đã lọc tại thời điểm xử lý mẫu, không phải dự đoán TCP robot.
- **Cách làm:** dự đoán theo vận tốc rồi sửa bằng sai số với phép đo. Confidence thấp làm tăng nhiễu đo trong
  mô hình, nên phép đo có ít ảnh hưởng hơn; gap dài hoặc thời gian không hợp lệ làm khởi tạo lại.
- **Giả định:** chuyển động giữa các mẫu không quá khác mô hình và timestamp có ý nghĩa.
- **Khi sai:** điểm mượt nhưng bám chậm, hoặc giật khi reset; cần so điểm trước/sau lọc và dt.
  Dự đoán nội bộ của Kalman không đồng nghĩa pipeline đang ngoại suy tới thời điểm robot tương lai.

#### 23.5.6. Retarget SP1/SP2: căn hướng, không tìm điểm gắp

- **Việc cần giải:** tìm góc khớp để hướng các đoạn tay robot giống hướng người.
- **Đầu vào:** `u`, `l`, `H`, offset hướng bàn tay, model robot và nghiệm tham chiếu `q_prev`.
- **Đầu ra:** 7 góc cùng sai số căn hướng/trạng thái clamp; chưa phải command motor.
- **Cách làm:** SP1 tìm một góc quay quanh một trục; SP2 tìm cặp góc liên quan hai trục. Các bước giải nhóm
  J1–J2, J3–J4, J5–J6 rồi J7; chọn nhánh phù hợp giới hạn và gần nghiệm trước.
- **Giả định:** quy ước trục/model đúng và quan sát các nhóm đủ tốt. Cụ thể code chỉ giải lại wrist khi cả
  `H` và `l` có giá trị; thấy được Hand không tự bảo đảm cẳng tay đã hợp lệ.
- **Khi sai:** sai hướng, đổi nhánh, chạm giới hạn hoặc giữ nghiệm cũ; xem `RetargetInfo`, dữ liệu nhóm và q
  trước/sau lọc. Sai số hướng nhỏ không chứng minh hai TCP robot có thể gặp nhau đúng vùng nhận vật.

#### 23.5.7. One Euro, deadband và xác nhận jump: ba chức năng khác nhau

- **Việc cần giải:** làm command đích bớt nhiễu nhưng vẫn tiếp nhận cử động có chủ đích.
- **Đầu vào:** 7 góc + độ mở kẹp thô, confidence và thời gian.
- **Đầu ra:** target đã lọc và cờ held cho từng phần tử.
- **Cách làm:** JointFilter gate dữ liệu kém/xác nhận bước nhảy trước; One Euro lọc thấp có cutoff thích ứng
  theo tốc độ ước lượng; deadband ngăn biến thiên nhỏ truyền ra đầu ra. Đây không phải ba tên của cùng một bộ lọc.
- **Giả định:** đơn vị đúng cho từng phần tử, thời gian đủ nhất quán và ngưỡng phù hợp biên độ tín hiệu/nhiễu.
- **Khi sai:** rung ít nhưng cử động nhỏ không qua hoặc bám chậm; xem lần lượt q thô, q lọc, held và command.
  Giảm deadband chỉ làm dễ truyền cử động nhỏ, không làm landmark chính xác hơn; giảm thời gian xác nhận jump
  có thể nhận nhiễu nhanh hơn. Kẹp dùng đơn vị 0..1, không dùng độ như J1–J7.

#### 23.5.8. GripMapper: chuyển cử chỉ sang độ mở

- **Việc cần giải:** đưa độ chụm hai ngón về miền điều khiển kẹp.
- **Đầu vào:** ratio quan sát hợp lệ và mốc pinch/open của người.
- **Đầu ra:** `g` trong 0..1 hoặc `NaN` khi quan sát không hợp lệ.
- **Cách làm:** chuẩn hoá tuyến tính rồi clip; nếu cấu hình `levels` thì thêm lượng tử hoá, hysteresis và dwell.
  Ví dụ minh hoạ `pinch=0,25`, `open=0,9`, ratio `0,575` cho `g=0,5` trước các tầng sau.
- **Giả định:** nhìn đúng tay/ngón và ratio còn có ý nghĩa khi đổi góc nhìn/nguồn.
- **Khi sai:** độ mở không như ý hoặc giữ giá trị cũ; xem ratio, confidence, nguồn và target/cmd/meas kẹp.
  Mapper không cảm nhận vật đã nằm trong má kẹp, không xác nhận lực giữ và không tự phối hợp hai tay nhả–nhận.

#### 23.5.9. SafetyGate: giới hạn bước lệnh, không bổ sung thông tin quan sát

- **Việc cần giải:** biến target thành chuỗi command theo điều kiện engage, thời gian và giới hạn hiện có.
- **Đầu vào:** target, fresh/held, thời gian, command trước và cấu hình.
- **Đầu ra:** command tiếp theo và trạng thái; backend tiếp tục kiểm tra trước gửi motor.
- **Cách làm:** kiểm tra quyền bám/dead-man, giới hạn góc/tốc độ/ramp và hình học capsule tay–tay. Capsule là
  đoạn thẳng có bán kính bao quanh, dùng làm xấp xỉ hình dạng link để tính khoảng hở.
- **Giả định:** model và trạng thái lệnh đủ hữu ích cho phép kiểm tra; không biết mọi vật cản ngoài mô hình.
- **Khi sai/không đạt:** command đứng hoặc chậm dù target đổi; xem trạng thái gate, limit/ramp và khoảng cách.
  Gate không sửa được ghép nhầm tay, không chứng minh pose thật trùng command và không thay kiểm chứng an toàn
  phần cứng. Collision hiện xét FK command, không phải hình học robot đo trực tiếp từ ảnh.

Nguồn cho các giải thích thuật toán: [multiview.py](openarm_shadow/fusion/multiview.py),
[handfusion.py](openarm_shadow/fusion/hand.py), [filters.py](openarm_shadow/filtering/joint.py),
[retarget.py](openarm_shadow/retarget/sew.py), [geometry.py](openarm_shadow/core/geometry.py),
[grip.py](openarm_shadow/retarget/grip.py), [safety.py](openarm_shadow/safety/gate.py).

**Cách nối các phần khi đọc lại báo cáo:** từ một biểu hiện ngoài đời, xác định nó xuất hiện từ tầng nào:
ảnh/landmark → điểm/hướng hợp nhất → nghiệm retarget → target lọc → command → feedback → TCP/vật thật.
Tầng đầu tiên xuất hiện sai lệch là nơi cần kiểm tra trước; một tầng phía sau làm đầu ra mượt hơn chưa chứng
minh đã sửa nguyên nhân ở tầng trước. Đây là cách đọc/chẩn đoán, không phải chỉ dẫn tự nới tham số để chạy thật.
