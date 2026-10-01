# Fusion 2 camera (webcam laptop + D435i)

Cấu hình hiện tại (`config/fusion_2cam.yaml`): **webcam laptop trực diện** (camera 0, khung tham chiếu, không có
depth) + **RealSense D435i lệch 45°** (có depth). D455 không dùng trong fusion.

Chạy `--source multi`: mỗi camera chạy MediaPipe Pose + Hand riêng, rồi các điểm vai, khuỷu, cổ tay và 21 điểm bàn tay
được **triangulate** trong một khung chung (khung camera đầu tiên). Depth của RealSense là bằng chứng phụ để phân xử
khi hai camera không khớp nhau. Phần retarget, bộ lọc và SafetyGate không đổi: fusion chỉ thay khâu nhận diện.

```
cam 0 (webcam laptop, trực diện) ─ luồng đọc ─┐                   ┌─ MediaPipe (cam 0) ─┐
                                               ├─ ghép khung theo ─┤                     ├─ triangulate có trọng số
cam 1 (D435i, lệch 45°, có depth) ─ luồng đọc ─┘   thời gian ≤25ms └─ MediaPipe (cam 1) ─┘  + depth phân xử + gate
                                                                                 │
                        retarget ◄── khung thân từ vai/hông 3D ◄── hướng bàn tay (tracker chống lật)
```

## Lấy gì từ đâu

Không chép code (tự viết lại theo ý tưởng); các repo được nhắc đến đều có license cho phép tham khảo.

| Ý tưởng | Nguồn | File |
| --- | --- | --- |
| Triangulate DLT có trọng số theo độ tin cậy của từng camera | Pose2Sim `weighted_triangulation` (BSD-3) | `multiview.triangulate_weighted` |
| Sai số chiếu lại lớn → bỏ camera tệ nhất rồi triangulate lại | Pose2Sim `triangulation_from_best_cameras` | `multiview.fuse_point` |
| Depth RealSense làm phương trình thứ 3, phân xử khi 2 camera mâu thuẫn | tự thêm (không repo nào làm cho 2 camera) | `multiview.fuse_point` |
| Bảng ChArUco, tư thế bảng ở từng camera → ngoại tham số tương đối, gộp nhiều lần chụp | caliscope (BSD-2), aniposelib (BSD-2), Pose2Sim `charuco_static` | `calibration.py`, `scripts/calibrate_cameras.py` |
| Mỗi camera một luồng đọc, ghép khung theo dấu thời gian gần nhất | stereohand (MIT) | `multiview.MultiCameraSource` |
| Camera nhìn lòng bàn tay rõ hơn thì trọng số lớn hơn | Fortini 2023 (orthogonality index), AnyTeleop (best view) | `MultiViewPerception._fuse_hand` |
| Nội tham số RealSense lấy từ SDK, deproject/project chuẩn | librealsense | `CameraModel` |
| Đặt camera: trực diện + ~45°; > 90° cho kết quả kém | Samani et al. 2026 | mục "Đặt camera" dưới đây |

Không dùng: OpenPose (chỉ phi thương mại), freemocap (AGPL), EasyMocap (dự án công ty phải đăng ký),
multical (LGPL). Tổng hợp đầy đủ: tài liệu "Multi-camera fusion" trong project.

## Đặt camera

- **front** (camera đầu tiên, khung tham chiếu): webcam laptop trực diện. Kê laptop (hộp, chồng sách) cho webcam
  cao ngang ngực, cách người 1,2–1,5 m; màn hình nghiêng sao cho webcam nhìn thẳng, không chúc xuống.
- **side45**: D435i lệch ~45° **về phía tay đang điều khiển** (tay phải), cùng độ cao, cùng khoảng cách, cắm cổng
  USB 3 (`list_cameras.py` cảnh báo nếu chạy USB 2).
- Cả hai phải thấy trọn vai, hông và tay trong cả vùng cử động.
- Sau khi hiệu chuẩn, **không gập/nghiêng màn hình laptop và không xê dịch laptop hay D435i**: webcam gắn vào màn
  hình, chỉnh góc màn hình là camera 0 đổi hướng. Chạm vào là phải hiệu chuẩn lại.
- Phòng đủ sáng: thiếu sáng thì webcam tự tăng thời gian phơi sáng, tụt xuống ~15 fps và ảnh nhoè khi tay cử động.
- Hiệu chuẩn xong thì **không xê dịch camera**. Chạm vào chân máy là phải hiệu chuẩn lại.

Không cần hiệu chuẩn camera ↔ robot: retarget dùng khung thân dựng từ vai và hông, nên chỉ cần hai camera
khớp nhau.

## Webcam laptop mờ: làm nét

Ảnh mờ thì MediaPipe đặt điểm lệch, nhất là 21 điểm bàn tay. Theo thứ tự, mỗi bước kiểm tra bằng
`python scripts/webcam_check.py --config config/fusion_2cam.yaml` (điểm nét trong khung vàng: cao hơn = nét hơn;
đặt bảng ChArUco cách webcam ~1,3 m và chỉ đổi một thứ mỗi lần):

1. **Lau ống kính webcam** bằng khăn mềm. Vết vân tay là nguyên nhân mờ hay gặp nhất.
2. **Độ phân giải là giới hạn phần cứng.** Webcam Latitude 5490 của nhóm (Integrated_Webcam_HD) chỉ có YUYV,
   tối đa 640x480 @ 30 fps (`v4l2-ctl -d /dev/video0 --list-formats-ext`), không có MJPG/720p. Bàn tay cách 1,3 m
   chỉ còn ~40 px. Webcam khác có MJPG 720p/1080p thì đặt `width/height`, `fourcc: MJPG` trong
   `fusion_2cam.yaml`; chương trình thử nhiều thứ tự đặt và báo `Cảnh báo: webcam ... không chạy được` nếu không
   nhận.
3. **Ánh sáng**: bật đủ đèn, chiếu vào người, không đứng ngược cửa sổ. Thiếu sáng -> webcam phơi sáng lâu -> tay
   cử động bị nhoè.
4. **Control của webcam** (`camera.v4l2`, cần `sudo apt install v4l-utils`): `power_line_frequency: 1` (50 Hz, đã
   bật), `sharpness` (0–6, mặc định 2), `auto_exposure: 1` + `exposure_time_absolute` (phơi sáng tay). Thử trực
   tiếp khi `webcam_check.py` đang chạy, ở terminal khác:
   `v4l2-ctl -d /dev/video0 --set-ctrl=sharpness=4` rồi xem điểm nét; giá trị tốt thì ghi vào config.
   `sharpness` là làm nét trong webcam: tăng vừa phải thôi, cao quá chỉ tăng viền và nhiễu (không thêm chi tiết).
5. Nếu vẫn mờ: thay webcam trực diện bằng webcam USB 1080p có MJPG (camera 0, `source:` = chỉ số mới), hoặc dùng
   lại D455 làm camera trực diện (`source: realsense` + serial cho cả hai RealSense).

**Đổi độ phân giải webcam thì phải hiệu chuẩn lại** (`calibrate_cameras.py` tự làm lại nội tham số khi độ phân giải
khác; `shadow.py` dừng và báo nếu không khớp).

## Các bước

```bash
source .venv/bin/activate

# 1) Xem chỉ số webcam laptop (dòng có tên kiểu "Integrated_Webcam_HD") và D435i.
#    Cắm D435i cũng tạo thêm /dev/video*, nên webcam laptop không chắc là 0: sửa `source:` của front nếu khác.
#    Chỉ có 1 RealSense nên serial của side45 để trống được.
python scripts/list_cameras.py

# 2) In bảng ChArUco: in 100% (không "fit to page"), dán lên tấm phẳng cứng.
#    ĐO LẠI cạnh 1 ô vuông bằng thước; khác 35 mm thì sửa fusion.board.square_m (và marker_m theo tỉ lệ).
python scripts/make_charuco_board.py -o charuco_a4.png

# 3) Hiệu chuẩn (1 lần, mỗi khi dời camera)
python scripts/calibrate_cameras.py --config config/fusion_2cam.yaml

# 4) Mô phỏng trước
python scripts/shadow.py --source multi --arms right --config config/fusion_2cam.yaml

# 5) Robot thật lần đầu với fusion (cổ tay khoá): first_real.yaml TRƯỚC, fusion_2cam.yaml SAU
python scripts/shadow.py --source multi --robot openarm --arms right \
    --config config/first_real.yaml --config config/fusion_2cam.yaml

# 6) Mở cổ tay, bám nhanh (vẫn giữ giới hạn tốc độ, tăng tốc mềm, chặn bước nhảy):
python scripts/shadow.py --source multi --robot openarm --arms right \
    --config config/wrist_real_30.yaml --config config/fusion_2cam.yaml --config config/fusion_real_fast.yaml

# 7) (tuỳ chọn) Bám theo vận tốc + gửi vận tốc xuống motor: thêm --config config/velocity_ff.yaml ở CUỐI.
#    Đo trước/sau: thêm --record run.npz rồi
python scripts/measure_lag.py run.npz
```

## Độ trễ và cách đo

Mô phỏng (camera 15 fps, giả định camera + nhận diện 120 ms) cho thấy trễ nằm ở:

| Khâu | Trễ | Đã xử lý |
| --- | --- | --- |
| Bộ lọc One Euro J1-J4 (beta 0.02) | 130-200 ms | fusion_real_fast: beta 0.5 -> ~70 ms; Pose nhiễu lớn thì đứng yên rung hơn, hạ về 0.2 nếu cần |
| Giới hạn tốc độ khi tay nhanh hơn giới hạn | rất lớn (vd 440 ms) | nâng dần trong fusion_real_fast.yaml |
| SafetyGate kiểu cũ: lao tới mục tiêu rồi đứng chờ khung sau | ~1/2 khung + giật theo nhịp 15 Hz | `velocity_tracking`: chạy đều theo vận tốc mục tiêu, giới hạn gia tốc |
| Motor nhận dq = 0: kd hãm chuyển động | kd/kp: ~70 ms cổ tay, ~40 ms J1 | `velocity_tracking.feedforward`: gửi vận tốc lệnh làm dq |

`config/velocity_ff.yaml` bật hai mục cuối (mô phỏng cổ tay ±40° 0,3 Hz: trễ mục tiêu -> motor 110 -> ~0 ms, sai số
5,8 -> 1,9°). Mất người/tay thì thôi ngoại suy ngay (không trôi tiếp), đứng yên không rung hơn kiểu cũ.
`--record run.npz` giờ ghi thêm lệnh và góc đo 100 Hz; `scripts/measure_lag.py run.npz` in trễ mục tiêu -> lệnh,
lệnh -> đo, mục tiêu -> đo cho từng khớp, để kiểm chứng trên robot thật.

## Chế độ nhẹ `body_source: front` (mặc định trong fusion_2cam.yaml)

Học từ bản Openarm_Teleop của nhóm (chạy mượt trên robot thật), giữ nguyên các lớp an toàn:

| | Cách làm | Tác dụng |
| --- | --- | --- |
| Cánh tay | Vai/khuỷu/cổ tay từ Pose của webcam (điểm world MediaPipe) | Giống chế độ 1 camera đã chạy ổn trên robot |
| Camera phụ | D435i chỉ chạy Hand (`pose: false`) | Bớt gần một nửa tính toán |
| Pose | Chạy 1/2 khung (`pose_interval: 2`), hụt <= 6 lần vẫn giữ kết quả cũ | Nhanh hơn, không mất tay khi Pose chớp |
| Bàn tay | Chỉ điều khiển 1 tay: mỗi camera tìm 1 bàn tay, luôn là tay đó; không dùng điểm handedness | Không mất tay khi mu/cạnh bàn tay quay về camera |
| Ghép khung | `pair_wait_s: 0`: không chờ camera phụ | Vòng lặp không khựng |
| Hướng bàn tay | `OrientationFusion`: 3D (Kabsch) + hướng từ điểm world từng camera + lòng bàn tay từ depth; 2 giả thuyết chống lật; đổi hướng > 100° cần 3 khung, trong lúc chờ cổ tay đứng yên | Không nhận sai hướng; xoay nhanh vẫn bám (làm mượt thích ứng) |

Không lấy từ bản kia: tắt giới hạn tốc độ (`velocity_limit_enabled: false`), `engage_blend_s: 0` (robot lao tới
mục tiêu trong 1 nhịp khi engage) và tắt chặn bước nhảy. `config/fusion_real_fast.yaml` thay vào đó nâng giới
hạn cổ tay lên 45°/s (J1-J4 30-40°/s), tăng tốc mềm 1 s, giữ bước nhảy 0,3 s. (Bản đầu để 90°/s / 0,15 s
làm cổ tay tự xoay nhanh khi mất hướng tay - xem mục "Cổ tay tự xoay nhanh".)

### Hiệu chuẩn (bước 3)

- Bảng hiện trên màn hình (thay giấy in) cũng dùng được, nhưng phải **đo cạnh ô trên màn hình** và sửa
  `fusion.board.square_m`, `marker_m` cho đúng. Sai kích thước ô thì góc giữa hai camera vẫn đúng nhưng khoảng cách
  bị co/giãn, depth D435i sẽ không khớp nghiệm 2D (hay hiện `!`). Giảm độ sáng phòng chiếu vào màn hình để bớt lóa.

- Cầm bảng trong vùng tay sẽ cử động, sao cho **cả hai** camera cùng thấy (mỗi camera ≥ 10 góc).
- Chương trình tự chụp khi bảng đã dời > 40 px và cách lần trước > 0,8 s. Đổi vị trí và nghiêng bảng giữa các lần;
  nghiêng vừa phải (±30°) để camera 45° vẫn thấy rõ.
- Đủ 20 lần thì tự tính; phím `c` tính sớm (khi ≥ 8 lần), `q` thoát không lưu.
- Kết quả in ra: góc lệch trục nhìn (nên ~45°), khoảng cách hai camera, sai số chiếu lại:
  < 3 px tốt · < 6 px tạm được · lớn hơn thì chụp lại. File ghi vào `config/cameras_calib.yaml`.
- **Webcam laptop cần hiệu chuẩn cả nội tham số** (tiêu cự, tâm ảnh, méo ống kính), vì không có số của nhà sản
  xuất như RealSense. Màn hình hiện `front noi tham so: N/20 anh`. Lúc đầu đưa bảng **sát webcam (40–60 cm)**,
  **nghiêng nhiều** (tới ±45° cả hai chiều) và đưa bảng qua các góc ảnh; lúc này D435i không cần thấy bảng. Sau đó
  mới lùi ra vùng tay để chụp chung cho hai camera. Chương trình chỉ tính khi đủ cả hai phần.
  - Giả lập: có pha nghiêng nhiều → lệch 0,4° / 0,5 cm; chỉ nghiêng ít → lệch 2,9° / 6,6 cm. Pha này quan trọng.
  - Kết quả in `front: nội tham số webcam từ N ảnh, sai số X px, góc nhìn ngang Y°`: sai số nên < 0,5 px;
    góc nhìn ngang webcam laptop thường 60–75°, lệch xa khoảng này là hiệu chuẩn hỏng.
  - Nội tham số webcam được lưu kèm độ phân giải. Lần hiệu chuẩn sau (vd chỉ dời D435i) tự dùng lại, chỉ cần chụp
    chung; thêm `--redo-intrinsics` nếu muốn làm lại. Khi chạy, nếu webcam cho ảnh khác độ phân giải lúc hiệu chuẩn
    thì chương trình dừng và báo.

## Đọc màn hình

Ảnh ghép các camera cạnh nhau; chấm tròn tím = điểm 3D sau fusion chiếu ngược lại vào camera 0 (nằm lệch khỏi khớp
là dấu hiệu hiệu chuẩn sai hoặc camera bị xê dịch). Các dòng chữ:

```
fusion 2 cam | lech khung 4 ms
right: vai 2cam 0px | khuyu 2cam 1px | co tay 2cam 0px D !      (chế độ triangulate)
right: vai/khuyu/co tay tu Pose camera 0                        (chế độ front)
ban tay 2cam 21/21 diem, nhin ro 0.99, TRACKING, khop long tay 3mm | nguon: 3d+rgb:front+depth:side45
```

- `lech khung`: độ lệch thời gian giữa hai khung đã ghép. Chờ tối đa `pair_wait_s` cho khung lệch <= `sync_tol_s`;
  khung phụ lệch > `max_skew_s` (40 ms) thì KHÔNG ghép lần đó (màn hình báo `MAT KHUNG`), để không triangulate hai
  tư thế khác thời điểm. Bình thường < 20 ms; thường xuyên > 30 ms là một camera đang rớt khung (USB 2, cáp kém).
- `2cam`/`1cam`: số camera dùng cho điểm đó; `Npx`: sai số chiếu lại lớn nhất, quy đổi về ảnh 640x480 để
  webcam 720p và D435i so được với nhau (1 px ~ 0,1°; tốt < 10 px).
- `D`: đã dùng depth (khớp nghiệm 2D, hoặc phân xử khi hai camera mâu thuẫn).
- `!`: depth nằm **xa hơn** nghiệm 2D → có mâu thuẫn không phân xử được; độ tin cậy điểm đó bị giảm một nửa.
- `nhin ro`: mức lòng bàn tay quay về phía camera tốt nhất (1 = nhìn thẳng, 0 = nhìn cạnh).
- Trạng thái hướng bàn tay: `TRACKING` >= 2 nguồn đồng ý · `DEGRADED` chỉ 1 nguồn (vẫn điều khiển) ·
  `SWITCH?` hướng mới cách xa, đang chờ xác nhận (cổ tay đứng yên) · `HOLD` mất tay, giữ hướng cũ, cổ tay đứng yên
  tối đa 12 khung · `ACQUIRE` vừa thấy lại tay (sau LOST/CONFLICT, hoặc sau HOLD mà hướng mới lệch > 45°): cổ tay
  đứng yên tới khi hướng ổn định (< 20°) 4 khung có ≥ 2 nguồn (1 nguồn: 8 khung) · `LOST` mất hẳn.
  `nguon:` các nguồn được dùng (3d, rgb:<camera>, depth:<camera>).

### J1/J3 nhảy lớn (khuỷu nhảy trước/sau thân) ở chế độ triangulate

Khi chỉ 1 camera thấy khuỷu, điểm khuỷu lấy từ ảnh + depth của D435i. Depth 1 điểm hay rơi vào thân/nền khi khuỷu
bị che, làm khuỷu nhảy ra trước/sau thân -> J1 và J3 đổi 90-110° trong 1 khung (đo được ở lần chạy thử đầu).
- `fusion.body_mono_depth_conf` (0.5): vai/khuỷu/cổ tay chỉ 1 camera + depth bị hạ độ tin cậy dưới
  `filter.min_conf` -> khớp giữ, không đi theo.
- `filter.jump_confirm_conf`: bước nhảy > `jump_deg` chỉ nhận khi giá trị mới ổn định suốt `jump_hold_s` và độ
  tin cậy đủ cao mỗi khung (J1-J4 0.7). Nhảy qua lại giữa hai nghiệm không bao giờ được nhận.
- `--record` giờ ghi thêm độ tin cậy từng khớp và số camera / sai số / depth của vai-khuỷu-cổ tay;
  `python scripts/find_jumps.py run.npz` liệt kê từng lần nhảy kèm chẩn đoán.

### Sau lần chạy robot thật đầu (cổ tay khóa)

Nhận diện tốt (vai/khuỷu/cổ tay 2 camera 99%, sai số 3-7 px; motor bám lệnh 80-120 ms, sai 1-3°). Còn 3 kiểu nhảy:
- J3 trôi tới giới hạn khi tay gần thẳng (trục J3 trùng trục J5 nên xoay cánh tay không quan sát được), gập khuỷu
  lại thì phải quay 147°. `retarget.elbow_j3_full_deg` (30): độ tin cậy J3 tăng dần từ 12° tới 30° gập khuỷu.
- J1 nhảy 71° khi khuỷu sai số 18 px. `fusion.body_err_conf_px` [10, 30]: điểm sai số lớn bị hạ độ tin cậy.
- J5 lật -87 -> 87° khi hướng tay chỉ 1 nguồn (DEGRADED 0.7). `filter.jump_confirm_conf` cổ tay 0.8: bước nhảy
  lớn cổ tay chỉ nhận khi >= 2 nguồn đồng ý.

### Cổ tay tự xoay nhanh khi tay để ngang

Bàn tay để ngang mà mép tay/đầu ngón chĩa vào camera thì lòng bàn tay gần như không nhìn thấy: MediaPipe đoán
hướng sai hoặc lật 180°, các nguồn mâu thuẫn → `HOLD`/`CONFLICT`. Bản cũ khi thấy lại tay nhận ngay hướng mới
(có thể sai/lật) rồi chạy tới đó ở 90°/s → cổ tay robot xoay vụt. Đã thêm 3 lớp:

1. `ACQUIRE` (trên): cổ tay không nhận hướng mới cho tới khi hướng ổn định nhiều khung.
2. SafetyGate `resume_after_s` / `resume_blend_s`: khớp nào đứng yên > 0,3 s rồi chạy lại thì tăng tốc mềm riêng
   khớp đó trong 1 s (J1-J4 không bị ảnh hưởng khi chỉ cổ tay mất).
3. `fusion_real_fast.yaml`: cổ tay 45°/s (trước 90), bước nhảy > 60° bị giữ 0,3 s.

Vẫn nên tránh tư thế mép tay chĩa thẳng vào cả hai camera: đặt camera phụ lệch 45-60° sao cho luôn có một camera
nhìn được lòng/mu bàn tay. Thấy `ACQUIRE`/`HOLD` lặp lại ở một tư thế = tư thế đó không quan sát được, đừng điều
khiển cổ tay ở tư thế đó.

Hiệu chuẩn tay tự động (tay thả xuôi, lòng bàn tay vào đùi) vẫn như cũ, tính theo camera đầu tiên.

## Cách hợp nhất một điểm (`fuse_point`)

1. Mỗi camera cho 2 phương trình tuyến tính theo toạ độ đã khử méo, nhân trọng số = visibility/độ tin cậy MediaPipe.
   Giải bình phương tối thiểu (DLT không thuần nhất).
2. ≥ 3 camera: sai số chiếu lại > `reproj_thresh_px` thì bỏ camera tệ nhất, lặp lại.
3. 2 camera mâu thuẫn: thử "camera A + depth của A" và "camera B + depth của B", chọn phương án chiếu vào camera
   còn lại khớp nhất; độ tin cậy × 0,5.
4. Hai camera khớp: depth trong `depth_consistency_m` (4 cm) thì thêm vào với trọng số `depth_weight`.
   Depth gần hơn = vật che (ngón che ngón), bỏ qua. Depth xa hơn = cờ `!`.
5. Chỉ 1 camera thấy: dùng được khi camera đó có depth (ở cấu hình này: chỉ D435i).

Hướng bàn tay: 21 điểm đã triangulate → mặt phẳng lòng bàn tay → `HandOrientationTracker` làm mượt, sửa dấu khi
cả hai camera nhìn cạnh bàn tay, và chỉ chấp nhận cú xoay > 100° khi nó kéo dài 4 khung liên tiếp.

## Giới hạn và việc còn lại

- **Chưa đo fps trên laptop thật với 2 camera.** MediaPipe chạy song song theo luồng, nhưng 2 × (Pose + Hand) trên
  CPU có thể < 15 fps. Nếu chậm: trong `config/fusion_2cam.yaml` đổi `models.pose` sang
  `models/pose_landmarker_lite.task` (đã có sẵn sau `download_models.sh`).
- Với đúng 2 camera, lỗi nằm **dọc đường epipolar** không phát hiện được bằng hình học (hai tia vẫn cắt nhau, chỉ
  sai độ sâu). Depth giúp được một phần (cờ `!`); camera thứ 3 mới giải quyết triệt để.
- Đồng bộ bằng phần mềm (dấu thời gian máy tính), không phải hardware sync. Webcam laptop thường trả khung **trễ
  hơn** RealSense vài chục ms; khi cử động nhanh, điểm 3D (nhất là cổ tay) bị kéo lệch. Có tuỳ chọn `latency_s` cho
  từng camera trong `fusion.cameras` (mặc định 0) để bù; chưa có công cụ đo tự động. Nếu tay đứng yên thì chấm tím
  khớp mà vẫy nhanh thì lệch và cột `px` tăng vọt, thử `latency_s: 0.03`–`0.06` cho front và so sánh.
- Camera 0 (webcam) không có depth: điểm chỉ webcam thấy (D435i bị che) thì không dùng được; điểm chỉ D435i thấy
  vẫn dùng được nhờ depth.
- Chế độ 1 camera `--source realsense` vẫn còn (dùng camera RealSense đang cắm, giờ là D435i); fusion không liên quan.
- Test giả lập (`tests/test_multiview.py`): triangulate, depth phân xử, cờ mâu thuẫn, chống lật lòng bàn tay khi xoay
  cổ tay 360°, hiệu chuẩn ChArUco (sai < 0,5° / 1 cm), luồng đầy đủ với camera giả, kiểm tra độ phân giải webcam.
  Chạy thử hiệu chuẩn với webcam méo ống kính làm camera 0: 0,4° / 0,5 cm. Chưa thử trên camera thật.

## An toàn khi chạy robot thật

Giống `docs/SAFETY.md`: luôn có người giữ E-stop, không ai trong tầm với của tay robot, tay robot thả xuôi trước
khi bật motor, chạy mô phỏng (bước 4) trước. Chỉ tay phải (`--arms right`); **tay trái chưa được bật** cho tới khi
hiệu chuẩn lại zero. Đo lại offset J4 mỗi buổi. Nếu màn hình hay hiện `!` hoặc chấm tím lệch khớp, dừng lại và
hiệu chuẩn camera lại trước khi engage.
