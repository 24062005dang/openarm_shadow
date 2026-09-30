# Fusion 2 camera (D455 + D435i)

Chạy `--source multi`: mỗi camera chạy MediaPipe Pose + Hand riêng, rồi các điểm vai, khuỷu, cổ tay và 21 điểm bàn tay
được **triangulate** trong một khung chung (khung camera đầu tiên). Depth của RealSense là bằng chứng phụ để phân xử
khi hai camera không khớp nhau. Phần retarget, bộ lọc và SafetyGate không đổi: fusion chỉ thay khâu nhận diện.

```
cam 0 (D455, trực diện) ─ luồng đọc ─┐                   ┌─ MediaPipe (cam 0) ─┐
                                      ├─ ghép khung theo ─┤                     ├─ triangulate có trọng số
cam 1 (D435i, lệch 45°) ─ luồng đọc ─┘   thời gian ≤25ms └─ MediaPipe (cam 1) ─┘  + depth phân xử + gate
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

- **front** (camera đầu tiên, khung tham chiếu): D455 trực diện, cao ngang ngực, cách người 1,2–1,5 m.
- **side45**: D435i lệch ~45° **về phía tay đang điều khiển** (tay phải), cùng độ cao, cùng khoảng cách.
- Cả hai phải thấy trọn vai, hông và tay trong cả vùng cử động. Cắm mỗi camera vào một cổng USB 3 riêng
  (`list_cameras.py` cảnh báo nếu camera chạy USB 2).
- Hiệu chuẩn xong thì **không xê dịch camera**. Chạm vào chân máy là phải hiệu chuẩn lại.

Không cần hiệu chuẩn camera ↔ robot: retarget dùng khung thân dựng từ vai và hông, nên chỉ cần hai camera
khớp nhau.

## Các bước

```bash
source .venv/bin/activate

# 1) Xem serial, điền vào config/fusion_2cam.yaml (thay SERIAL_D455, SERIAL_D435I)
python scripts/list_cameras.py

# 2) In bảng ChArUco: in 100% (không "fit to page"), dán lên tấm phẳng cứng.
#    ĐO LẠI cạnh 1 ô vuông bằng thước; khác 35 mm thì sửa fusion.board.square_m (và marker_m theo tỉ lệ).
python scripts/make_charuco_board.py -o charuco_a4.png

# 3) Hiệu chuẩn (1 lần, mỗi khi dời camera)
python scripts/calibrate_cameras.py --config config/fusion_2cam.yaml

# 4) Mô phỏng trước
python scripts/shadow.py --source multi --arms right --config config/fusion_2cam.yaml

# 5) Robot thật: first_real.yaml TRƯỚC, fusion_2cam.yaml SAU (file sau ghi đè file trước)
python scripts/shadow.py --source multi --robot openarm --arms right \
    --config config/first_real.yaml --config config/fusion_2cam.yaml
```

### Hiệu chuẩn (bước 3)

- Cầm bảng trong vùng tay sẽ cử động, sao cho **cả hai** camera cùng thấy (mỗi camera ≥ 10 góc).
- Chương trình tự chụp khi bảng đã dời > 40 px và cách lần trước > 0,8 s. Đổi vị trí và nghiêng bảng giữa các lần;
  nghiêng vừa phải (±30°) để camera 45° vẫn thấy rõ.
- Đủ 20 lần thì tự tính; phím `c` tính sớm (khi ≥ 8 lần), `q` thoát không lưu.
- Kết quả in ra: góc lệch trục nhìn (nên ~45°), khoảng cách hai camera, sai số chiếu lại:
  < 3 px tốt · < 6 px tạm được · lớn hơn thì chụp lại. File ghi vào `config/cameras_calib.yaml`.
- Webcam (không phải RealSense) được hiệu chuẩn luôn nội tham số từ chính các ảnh bảng. Khi đó cần **nghiêng bảng
  nhiều** (tới ±45°) và phủ cả góc ảnh; thử giả lập với ít góc nghiêng cho lệch 2,9° / 6,6 cm. RealSense dùng nội
  tham số của nhà sản xuất nên không bị vấn đề này.

## Đọc màn hình

Ảnh ghép các camera cạnh nhau; chấm tròn tím = điểm 3D sau fusion chiếu ngược lại vào camera 0 (nằm lệch khỏi khớp
là dấu hiệu hiệu chuẩn sai hoặc camera bị xê dịch). Các dòng chữ:

```
fusion 2 cam | lech khung 4 ms
right: vai 2cam 0px | khuyu 2cam 1px | co tay 2cam 0px D !
ban tay 2cam 21/21 diem, nhin ro 0.99, FUSED
```

- `lech khung`: độ lệch thời gian giữa hai khung đã ghép (chờ tối đa `sync_tol_s` = 25 ms cho khung khớp, không có
  thì lấy khung gần nhất). Bình thường < 20 ms; thường xuyên > 30 ms là một camera đang rớt khung (USB 2, cáp kém).
- `2cam`/`1cam`: số camera dùng cho điểm đó; `Npx`: sai số chiếu lại lớn nhất (tốt < 10 px).
- `D`: đã dùng depth (khớp nghiệm 2D, hoặc phân xử khi hai camera mâu thuẫn).
- `!`: depth nằm **xa hơn** nghiệm 2D → có mâu thuẫn không phân xử được; độ tin cậy điểm đó bị giảm một nửa.
- `nhin ro`: mức lòng bàn tay quay về phía camera tốt nhất (1 = nhìn thẳng, 0 = nhìn cạnh).
- Trạng thái hướng bàn tay: `FUSED` bình thường · `SIGN-FIX` cả hai camera nhìn cạnh bàn tay, đang giữ dấu úp/ngửa
  theo khung trước · `HOLD` mất tay, giữ hướng cũ tối đa 8 khung · `NONE` không có.

Hiệu chuẩn tay tự động (tay thả xuôi, lòng bàn tay vào đùi) vẫn như cũ, tính theo camera đầu tiên.

## Cách hợp nhất một điểm (`fuse_point`)

1. Mỗi camera cho 2 phương trình tuyến tính theo toạ độ đã khử méo, nhân trọng số = visibility/độ tin cậy MediaPipe.
   Giải bình phương tối thiểu (DLT không thuần nhất).
2. ≥ 3 camera: sai số chiếu lại > `reproj_thresh_px` thì bỏ camera tệ nhất, lặp lại.
3. 2 camera mâu thuẫn: thử "camera A + depth của A" và "camera B + depth của B", chọn phương án chiếu vào camera
   còn lại khớp nhất; độ tin cậy × 0,5.
4. Hai camera khớp: depth trong `depth_consistency_m` (4 cm) thì thêm vào với trọng số `depth_weight`.
   Depth gần hơn = vật che (ngón che ngón), bỏ qua. Depth xa hơn = cờ `!`.
5. Chỉ 1 camera thấy: dùng được khi camera đó có depth (như chế độ D455 đơn).

Hướng bàn tay: 21 điểm đã triangulate → mặt phẳng lòng bàn tay → `HandOrientationTracker` làm mượt, sửa dấu khi
cả hai camera nhìn cạnh bàn tay, và chỉ chấp nhận cú xoay > 100° khi nó kéo dài 4 khung liên tiếp.

## Giới hạn và việc còn lại

- **Chưa đo fps trên laptop thật với 2 camera.** MediaPipe chạy song song theo luồng, nhưng 2 × (Pose + Hand) trên
  CPU có thể < 15 fps. Nếu chậm: trong `config/fusion_2cam.yaml` đổi `models.pose` sang
  `models/pose_landmarker_lite.task` (đã có sẵn sau `download_models.sh`).
- Với đúng 2 camera, lỗi nằm **dọc đường epipolar** không phát hiện được bằng hình học (hai tia vẫn cắt nhau, chỉ
  sai độ sâu). Depth giúp được một phần (cờ `!`); camera thứ 3 mới giải quyết triệt để.
- Đồng bộ bằng phần mềm (dấu thời gian máy tính), không phải hardware sync. Cử động rất nhanh có thể lệch vài cm.
- Hai RealSense chiếu IR cùng lúc có thể nhiễu depth của nhau khi nhìn cùng vùng; thường chấp nhận được với D4xx,
  nếu depth lỗ chỗ thì tắt emitter một camera.
- Test giả lập (`tests/test_multiview.py`): triangulate, depth phân xử, cờ mâu thuẫn, chống lật lòng bàn tay khi xoay
  cổ tay 360°, hiệu chuẩn ChArUco (sai < 0,5° / 1 cm), luồng đầy đủ với camera giả. Chưa thử trên camera thật.

## An toàn khi chạy robot thật

Giống `docs/SAFETY.md`: luôn có người giữ E-stop, không ai trong tầm với của tay robot, tay robot thả xuôi trước
khi bật motor, chạy mô phỏng (bước 4) trước. Chỉ tay phải (`--arms right`); **tay trái chưa được bật** cho tới khi
hiệu chuẩn lại zero. Đo lại offset J4 mỗi buổi. Nếu màn hình hay hiện `!` hoặc chấm tím lệch khớp, dừng lại và
hiệu chuẩn camera lại trước khi engage.
