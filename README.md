# openarm_shadow

Teleop OpenArm v1.0 bằng camera 2D: robot bắt chước dáng tay người (shadowing).

Repo gồm hai phần: **code** chạy được (mô phỏng + OpenArm thật qua CAN-FD) và **tài liệu** nhóm đã tổng hợp
(phân tích đề tài, thông số OpenArm v1.0, bring-up, phân tích 5 repo, bản dịch 2 bài báo, tài liệu tham khảo).
Bắt đầu từ [`docs/README.md`](docs/README.md).

```
Webcam / điện thoại ──► MediaPipe Pose + Hand ──► khung thân người ──► retarget kiểu SEW-Mimic ──► lọc từng khớp
                                                                                  │
                         OpenArm v1.0 (CAN-FD) ◄── robot backend ◄── SafetyGate ◄──┘
                         hoặc robot mô phỏng: hình que (sim) / MuJoCo 3D (mujoco)
```

| Khâu | File | Lấy ý tưởng từ |
| --- | --- | --- |
| Nhận diện người + bàn tay (CPU) | `openarm_shadow/vision/perception.py` | MediaPipe Tasks; ghép bàn tay theo cổ tay như repo Marionette |
| Khung thân (x trước, y trái, z lên) | `geometry.make_frame` | MakeFrame trong SEW-Mimic |
| Ánh xạ sang 7 khớp | `openarm_shadow/mapping/retarget.py` | SEW-Mimic (arXiv 2602.01632): căn **hướng** cánh tay, cẳng tay, bàn tay bằng bài toán con SP1/SP2 |
| Lọc | `openarm_shadow/filtering/filters.py` | One Euro + vùng chết + bỏ bước nhảy (Marionette), EMA điểm mốc 0.8 (Hand Shadowing) |
| An toàn | `openarm_shadow/safety/gate.py` | Ly hợp + dead-man + tăng tốc mềm (Marionette, kinetic-arm-lab), capsule chống hai tay va nhau (SEW-Mimic) |
| Fusion 2 camera (`--source multi`) | `openarm_shadow/fusion/` (triangulate, MultiViewPerception), `camera/calibration.py` | Pose2Sim (triangulate có trọng số), caliscope/aniposelib (ChArUco), stereohand (đồng bộ) — xem `docs/FUSION.md` |
| Robot thật | `openarm_shadow/robot/openarm_can_robot.py` | `openarm_can`, gain chính thức v1.0 |
| Robot mô phỏng MuJoCo (`--robot mujoco`) | `openarm_shadow/robot/mujoco_robot.py` + `openarm_mujoco/v1/` | MJCF của `enactic/openarm_mujoco` (góc khớp = góc URDF, FK lệch 0 mm) |
| Một tay robot (trái / phải dùng chung) | `openarm_shadow/mapping/arm.py` | Lớp `Arm`: động học, bộ giải, bộ lọc, kẹp, hiệu chuẩn của một tay; `MIRROR_SIGNS` (J1,J2,J3,J5,J6,J7 đảo dấu, J4 giữ) |
| Động học OpenArm v1.0 | `openarm_shadow/core/kinematics.py` + `data/openarm_v10_arms.json` | URDF của `enactic/openarm_description` |

Toàn bộ code tự viết; không chép code từ các repo trên (phần lớn không có license).

Repo này chỉ cho **teleop thời gian thực bằng camera**. Bài múa Thái Cực (phát lại quỹ đạo mocap đã tính sẵn,
`taichi_player`) là một hướng riêng của nhóm, không nằm ở đây. Hai repo chỉ dùng chung robot, bring-up CAN và cách
lọc số đọc rác.

## Trạng thái

- Đã kiểm tra (trên máy không có camera/robot): động học, retarget (thử ngược 500 tư thế, sai lệch < 1e-10°),
  bộ lọc, SafetyGate, luồng chạy sim với dữ liệu người giả lập, backend CAN với `openarm_can` giả lập.
  `pytest` 167/167 đạt (gồm test lọc số đọc rác của backend CAN, fusion 2 camera, làm sạch bàn tay giả lập và
  backend MuJoCo).
- `scripts/demo_sim.py` chạy trọn luồng pipeline → SafetyGate → robot mô phỏng với người giả lập (đã chạy được,
  cả hình que lẫn MuJoCo).
- Robot MuJoCo (`--robot mujoco`, 05/10): chạy được với `demo_sim.py` và webcam laptop. Mô hình MJCF v1 trùng
  động học `core/kinematics.py`; giới hạn J4 của MJCF hẹp hơn URDF (128-131°) nên dùng kèm `config/mujoco_sim.yaml`.
- Đã chạy với webcam thật trên laptop của nhóm (mô phỏng, 28/09): nhận diện và bám theo tay.
- Tay phải đã chạy trên OpenArm thật (30/09: J1–J4; 02/10: J1–J7 + kẹp, 2 camera). Tay trái: không phải zero sai mà
  J1/J2 lệch 180° (docs/SAFETY.md), offset ở `config/both_arms_real.yaml`, CHƯA chạy thật.
  Vẫn làm theo `docs/SAFETY.md` mỗi buổi (dry-run trước).
- Quy ước góc URDF ↔ motor: dấu đồng nhất; offset zero đo trên robot nằm trong `urdf_to_motor` của các config
  chạy thật (J4 tay phải trôi giữa các buổi: **đo lại mỗi buổi**).
- Fusion 2 camera (webcam + D435i, `--source multi`): chạy được trên máy nhóm; độ chính xác còn phụ thuộc webcam.

## Cài đặt (Ubuntu 24.04, Python 3.12, không cần GPU)

```bash
cd openarm_shadow
python3 -m venv --system-site-packages .venv   # system-site để thấy python3-openarm-can cài bằng apt
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_models.sh                # model MediaPipe vào models/
python -m pytest -q                            # 167 test phải đạt (test MuJoCo tự bỏ qua nếu chưa cài mujoco)
python scripts/check_kinematics.py             # in trục khớp, thử ngược retarget
```

Robot thật cần thêm `openarm_can` (xem `tools/bringup/`): `sudo apt install python3-openarm-can`.
RealSense D455 cần thêm: `pip install -e '.[realsense]'` (SDK đã thử với `pyrealsense2` 2.58.1).
Robot mô phỏng MuJoCo cần thêm: `pip install -e '.[mujoco]'` (đã thử `mujoco` 3.14) và mô hình ở `openarm_mujoco/v1/`.

## Chạy

```bash
# 0) Mô phỏng không cần camera, model hay robot: người giả lập làm vài động tác
python scripts/demo_sim.py                      # q/Esc để thoát; --out demo.mp4 để ghi video
python scripts/demo_sim.py --robot mujoco --config config/mujoco_sim.yaml   # thêm cửa sổ MuJoCo 3D

# 1) Mô phỏng với camera + hình que robot. Luôn chạy bước này trước robot thật.
python scripts/shadow.py                        # mặc định: RealSense (camera.index trong config)
python scripts/shadow.py --mode mirror          # đứng đối diện robot, như soi gương
python scripts/shadow.py --arms right           # chỉ điều khiển tay phải
python scripts/shadow.py --source realsense      # D455 duy nhất: RGB MediaPipe + depth metric
python scripts/shadow.py --source 0              # webcam laptop: không có depth, hướng tay từ MediaPipe
python scripts/shadow.py --source multi --config config/fusion_3cam.yaml     # 3 camera, 2 tay (cách dùng chính, xem docs/FUSION.md)

# 1b) Như bước 1 nhưng robot là OpenArm v1 trong MuJoCo (vật lý, gain v1.0, bù trọng lực) thay hình que.
#     config/mujoco_sim.yaml luôn đặt SAU CÙNG. Thêm --robot mujoco vào bất kỳ lệnh nào ở bước 1.
python scripts/shadow.py --robot mujoco --config config/mujoco_sim.yaml                      # D455
python scripts/shadow.py --robot mujoco --source 0 --arms right --config config/mujoco_sim.yaml  # webcam laptop
python scripts/shadow.py --robot mujoco --source multi \
    --config config/fusion_3cam.yaml --config config/mujoco_sim.yaml                         # 3 camera, 2 tay
# 3 camera (webcam + D455 + D435i): đủ các bước kiểm tra camera, in bảng, hiệu chuẩn, mô phỏng, robot thật trong
# docs/FUSION.md mục "Fusion 3 camera". Tóm tắt:
#   python scripts/list_cameras.py
#   python scripts/make_charuco_board.py --config config/fusion_3cam.yaml -o charuco_3cam_a4.png
#   python scripts/calibrate_cameras.py --config config/fusion_3cam.yaml --redo-intrinsics
#   python scripts/shadow.py --source multi --config config/fusion_3cam.yaml

# 2) Chế độ offline: video quay sẵn -> quỹ đạo (thử pipeline khi chưa có robot, thu demo cho IL)
python scripts/offline_retarget.py demo.mp4 -o demo.npz --show
python scripts/replay_npz.py demo.npz            # xem lại trên robot mô phỏng
python scripts/replay_npz.py demo.npz --robot mujoco --config config/mujoco_sim.yaml   # xem lại trong MuJoCo

# 3) OpenArm thật (làm theo docs/SAFETY.md)
./tools/bringup/setup_can.sh                                        # bật can0 + can1 (1 Mbps / 5 Mbps CAN-FD); mỗi lần cắm lại USB-CAN hoặc khởi động lại máy
#   ./tools/bringup/setup_can.sh can0                               # chỉ một cổng
#   ip -br link | grep can                                          # cả hai phải UP;  candump -n 20 can0  (bật nguồn robot trước)
python tools/bringup/read_joints.py --iface can0                    # chỉ đọc góc: can0 = tay phải, can1 = tay trái
python scripts/shadow.py --robot openarm --dry-run                  # motor TẮT: chỉ đọc, kiểm tra chiều khớp
python scripts/shadow.py --source multi --robot openarm --arms right \
    --config config/first_real.yaml --config config/fusion_3cam.yaml            # lần đầu: J1–J4, chậm
python scripts/shadow.py --source multi --robot openarm \
    --config config/wrist_real_30.yaml --config config/both_arms_real.yaml \
    --config config/fusion_3cam.yaml                                            # hai tay (sau khi dry-run đúng chiều)
# tuỳ chọn, thêm CUỐI lệnh: config/real_tracking.yaml (bám liên tục giữa 2 khung camera),
#                           config/auto_engage_real.yaml (tự đồng bộ khi READY đủ giây)
```

Hướng bàn tay tự hiệu chuẩn khi tay thả xuôi, xoè bàn tay, **lòng bàn tay nhìn camera** và đứng yên khoảng 0,6 s; màn hình báo
`Auto calib ... OK`. Phím khi chạy: `SPACE` engage / nhả (ly hợp) · `c` hiệu chuẩn lại thủ công · `b` học lại khung
thân · `p` về tư thế nghỉ · `q`/`Esc` về tư thế nghỉ rồi thoát.

Tham chiếu thân (`orientation.body_ref`, mặc định bật): lúc đầu đứng thẳng, thả tay xuôi khoảng 1 s để học tham
chiếu thân (màn hình `than: theo do`). Chưa học xong thì chưa READY / chưa engage. Sau đó mỗi khung so vai/hông đo
được với tham chiếu, lọc theo **độ nhất quán** chứ không theo độ tin cậy (tay che thân thì MediaPipe vẫn báo vai/hông
"thấy rõ" nhưng đặt sai chỗ):

- lệch ít (vai < 3 cm, hông < 2 cm): dùng điểm đo, tham chiếu trôi chậm theo, nên cử động thật nhỏ vẫn qua;
- lệch nhiều (vai > 8 cm, hông > 6 cm): coi là bị che, dùng tham chiếu (`than: uoc luong N diem`); ở giữa trộn mềm;
- cả thân dịch như một khối cứng liên tục 1 s (bước sang, xoay người): nhận vị trí mới. Chỉ vài điểm lệch, thân méo
  (tay che) thì không nhận.

Fusion triangulate lọc từng điểm vai/hông rồi dựng lại khung thân; 1 camera chỉ lọc hướng khung thân. Vai bị che hẳn
vẫn chạy tiếp bằng tham chiếu. Ảnh camera (fusion) vẽ thân đang dùng (xanh ngọc; điểm đang ước lượng: vòng cam) và
cánh tay hợp nhất (tím) thay vai/hông thô; khung xương xanh lá còn lại là kết quả thô của từng camera.
Khuỷu / cổ tay (khi bật `filter.landmark_kalman`, có sẵn trong config 3 camera): điểm đo vọt xa dự đoán Kalman thì bị
loại, dùng dự đoán đặt lại đúng độ dài xương đã học; lệch liên tục vài khung thì nhận (chuyển động nhanh thật).
Phím `b`: học lại tham chiếu thân (và hiệu chuẩn lại bàn tay). Tắt: `orientation.body_ref.enabled: false`.

Camera: **3 camera, 2 tay** = webcam laptop ở giữa (trực diện) + RealSense D455 lệch 45° phía tay phải + RealSense D435i
lệch 45° phía tay trái (2 camera có depth). Làm theo [`docs/FUSION.md`](docs/FUSION.md) mục "Fusion 3 camera"
(`config/fusion_3cam.yaml`, hiệu chuẩn `config/cameras_calib_3cam_rs.yaml`). Chạy 1 camera (`--source 0` / `realsense`)
vẫn được để thử nhanh, nhưng không phải cấu hình đang dùng.

Dùng robot MuJoCo (`--robot mujoco`): thay robot "lý tưởng" (đo = lệnh) bằng OpenArm v1 trong MuJoCo. Vẫn đi qua
đúng pipeline và SafetyGate như robot thật, chỉ khác backend (`openarm_shadow/robot/mujoco_robot.py`). Có hai cửa sổ:
cửa sổ OpenCV như cũ (camera + hình que; nét **xanh lá** = góc thật trong MuJoCo, để thấy robot trễ / võng so với
lệnh) và cửa sổ MuJoCo 3D (chạy ở tiến trình riêng, chỉ để xem: kéo robot bằng chuột không tác động vào mô phỏng).
Như chế độ sim: giữ READY 3 s là tự engage, không hỏi `yes`. Tuỳ chọn trong `robot.mujoco` (`config/default.yaml`):
`mode: physics` (actuator kp/kv như gain v1.0, mặc định) hoặc `kinematic` (robot đi đúng lệnh), `gravity_comp`,
`viewer`. `config/mujoco_sim.yaml` hạ giới hạn mềm J4 xuống 128° cho vừa giới hạn khớp của MJCF.

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
- Hướng kẹp: hai ngón kẹp đóng / mở theo **pháp tuyến lòng bàn tay** (mặt phẳng hai ngón kẹp vuông góc lòng bàn tay,
  như ngón cái và ngón trỏ khi gắp). Tay xuôi, ngón cái ra trước (hoặc cẳng tay ra trước, ngón cái hướng lên) ⟷
  J5 = 0, giữa tầm xoay; ngửa / úp hết cỡ ⟷ J5 = ±90°. Tư thế hiệu chuẩn (lòng bàn tay nhìn camera) ⟷ J5 = +90° tay
  phải, −90° tay trái (`Arm.calib_wrist`).
- Kẹp: r = (đầu ngón cái − đầu ngón trỏ) / chiều dài bàn tay, chụm = đóng (0), xoè = mở hẳn (1), mở liên tục theo
  hai ngón (`openarm_shadow/mapping/grip.py`). Phím **g** khi chạy: chụm hết cỡ rồi xoè hết cỡ trong 4 s để đo theo tay
  mình (số in ra terminal, ghi vào `grip:` để dùng lần sau). Chỉ muốn vài mức: `grip.levels: [0, 0.5, 1]`.
  Motor kẹp chỉ chạy khi `robot.gripper.enabled: true` (đo trước góc mở/đóng thật bằng `tools/bringup/read_joints.py`).

Chi tiết và nguồn: `docs/DESIGN.md`, `docs/05_diem_moi_2_bai_bao.md`.

## Cấu trúc mã nguồn (`openarm_shadow/`)

| Gói | Nội dung | File chính |
| --- | --- | --- |
| `core/` | hình học, động học, kiểu dữ liệu dùng chung (không phụ thuộc camera / robot) | `geometry.py`, `kinematics.py`, `rotation.py`, `types.py` (`ArmObs`, `Frame`) |
| `camera/` | nguồn khung webcam / RealSense, nhiều camera, mô hình camera, hiệu chuẩn ChArUco | `sources.py`, `multi_source.py`, `model.py`, `calibration.py` |
| `vision/` | MediaPipe Pose + Hand, depth RealSense, khung thân, ghép bàn tay, luồng nhận diện nền | `perception.py`, `depth.py`, `body.py` (`BodyRef`), `handfusion.py`, `landmarks.py`, `worker.py` |
| `fusion/` | hợp nhất nhiều camera: triangulate, hướng bàn tay chống lật, `MultiViewPerception` | `triangulation.py`, `orientation.py`, `multiview.py` |
| `mapping/` | từ quan sát tay người tới góc khớp | `retarget.py`, `arm.py` (`Arm`, `MIRROR_SIGNS`), `grip.py`, `pipeline.py` (`ShadowPipeline`) |
| `filtering/` | One Euro, vùng chết, Kalman điểm 3D, độ dài xương | `filters.py` |
| `safety/` | `SafetyGate`: giới hạn khớp / vận tốc, ly hợp, dead-man, chống va chạm | `gate.py` |
| `control/` | luồng điều khiển robot `Controller`, về tư thế nghỉ `park` | `controller.py` |
| `robot/` | backend robot: mô phỏng, MuJoCo, OpenArm thật qua CAN-FD | `sim.py`, `mujoco_robot.py`, `openarm_can_robot.py` |
| `display/` | khung xương, hình que robot, dòng chẩn đoán trên màn hình | `viz.py`, `overlay.py` |
| (gốc gói) | `app.py` vòng chạy chính nối các gói trên, `config.py` đọc config | |

Luồng dữ liệu: `camera` -> `vision` (-> `fusion` nếu nhiều camera) -> `mapping` (dùng `filtering`) -> `safety` ->
`control` -> `robot`; `display` vẽ kết quả. Phụ thuộc giữa các gói (đã kiểm tra, không có vòng): `core` không import gói
nào; `camera`, `vision`, `safety`, `mapping` chỉ dùng `core` (`mapping` thêm `filtering`); `fusion` dùng `camera`, `vision`,
`core` và `display` (để vẽ ảnh nhiều camera); `app.py` nối tất cả.

## Cấu trúc

```
config/default.yaml          mọi tham số (ghi đè bằng --config file_của_bạn.yaml)
openarm_shadow/              thư viện
scripts/demo_sim.py          mô phỏng không cần camera (người giả lập)
scripts/shadow.py            chạy teleop (sim / openarm)
scripts/offline_retarget.py  video -> .npz
scripts/replay_npz.py        phát .npz qua SafetyGate
scripts/check_kinematics.py  kiểm tra động học + retarget
scripts/extract_kinematics.py sinh lại data JSON từ URDF
scripts/list_cameras.py      liệt kê RealSense (serial, USB) và webcam
scripts/webcam_check.py      độ nét, fps, định dạng và control của webcam
scripts/measure_lag.py       đo độ trễ mục tiêu -> lệnh -> góc đo từ file --record
scripts/find_jumps.py        liệt kê các lần mục tiêu khớp nhảy lớn trong file --record, kèm số camera / sai số / depth
scripts/bench_mediapipe.py   so tốc độ MediaPipe CPU và GPU trên máy này (models.delegate)
scripts/make_charuco_board.py in bảng ChArUco A4
scripts/calibrate_cameras.py hiệu chuẩn ngoại tham số nhiều camera -> config/cameras_calib.yaml
tools/bringup/               script bật CAN, đọc khớp, lắc J7 (bring-up robot)
config/mujoco_sim.yaml       config cho --robot mujoco (đặt sau cùng)
openarm_mujoco/v1/           MJCF + mesh OpenArm v1 (enactic/openarm_mujoco, Apache-2.0) cho --robot mujoco
tests/                       pytest
docs/README.md               mục lục tài liệu + lộ trình
docs/01..07_*.md             phân tích đề tài, OpenArm v1.0, bring-up, dữ liệu, 2 bài báo, tham khảo
docs/SAFETY.md, DESIGN.md    checklist an toàn, thiết kế code
docs/FUSION.md               fusion 3 camera: đặt camera, hiệu chuẩn, chạy, đọc màn hình, giới hạn
docs/research/               phân tích 5 repo, bản dịch SEW-Mimic và Hand Shadowing
```
