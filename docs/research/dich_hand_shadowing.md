# Bản dịch: Vision-Based Hand Shadowing

Bài gốc: *Vision-Based Hand Shadowing for Robotic Manipulation via Inverse Kinematics* (Bắt chước tay bằng thị giác cho thao tác robot qua động học ngược), [arXiv 2603.11383v2](https://arxiv.org/html/2603.11383v2). Tác giả: Hendrik Chiche, Antoine Jamme (OMGrab Inc. và UC Berkeley), Trevor Rigoberto Martinez, Gabriel Gomes (UC Berkeley). Code: [chichonnade/Vision-Based-Hand-Shadowing](https://github.com/chichonnade/Vision-Based-Hand-Shadowing).

Dịch theo đúng thứ tự các mục của bản HTML trên arXiv. Hình vẽ không chép lại, chỉ dịch chú thích. Công cụ đọc web của mình trả về văn bản bài báo, đôi chỗ đã rút gọn nhẹ câu chữ; các con số và công thức được giữ nguyên. **Lưu ý trước khi đọc:** pipeline của bài này chạy **offline** (quay video trước, xử lý sau), không phải teleop thời gian thực, và camera đặt trên kính đeo mắt chứ không đặt ngoài.

## Tóm tắt (Abstract)

Teleop tay máy giá rẻ vẫn khó vì khó ánh xạ chuyển động bàn tay người sang lệnh khớp robot. Chúng tôi trình bày một pipeline retargeting **offline** kiểu "bàn tay chiếu bóng" (hand-shadowing) bằng động học ngược (IK), dùng một camera RGB-D góc nhìn thứ nhất (egocentric) gắn trên kính in 3D. Pipeline phát hiện 21 điểm mốc mỗi bàn tay bằng MediaPipe Hands, chiếu ngược (deproject) sang 3D nhờ cảm biến độ sâu, đổi sang hệ toạ độ robot, rồi giải bài toán IK bình phương tối thiểu có giảm chấn (damped least squares) để ra lệnh khớp cho robot SO-ARM101 (5 khớp tay + 1 khớp kẹp). Bộ điều khiển kẹp ánh xạ hình học ngón cái–ngón trỏ sang độ mở kẹp, có nhiều mức dự phòng. Hành động được xem trước trong mô phỏng vật lý rồi mới phát lại trên robot thật.

Chúng tôi đánh giá trên bài kiểm tra gắp–đặt có cấu trúc (lưới 5 ô, 10 lần gắp mỗi ô, 3 lần chạy độc lập), đạt tỉ lệ thành công **86,7% ± 4,2%**, và so với bốn chính sách thị giác–ngôn ngữ–hành động (VLA) — ACT, SmolVLA, π₀.₅, GR00T N1.5 — huấn luyện từ dữ liệu teleop tay dẫn–tay theo. Chúng tôi phân tích sai số định lượng: sai số vị trí IK trung bình **36,4 mm**; chỉ số độ mượt cho thấy lọc EMA giảm độ giật (jerk) **57–68%**; và thí nghiệm bỏ bớt trên tham số làm mượt. Thử trong môi trường thực không cấu trúc (siêu thị, hiệu thuốc), tỉ lệ thành công giảm còn **9,3%** do bàn tay bị vật xung quanh che khuất. Để khắc phục, chúng tôi tích hợp WiLoR làm bộ phát hiện tay thay thế, tăng tỉ lệ phát hiện tay thêm 8% so với MediaPipe — cho thấy cả triển vọng lẫn hạn chế hiện tại của retargeting giải tích không marker.

## I. Giới thiệu

Dạy robot thao tác vật như con người là mục tiêu trung tâm của ngành robot. Hai hướng chính đã hình thành: *teleop*, nơi người vận hành điều khiển trực tiếp robot theo thời gian thực, và *học bắt chước*, nơi một chính sách được huấn luyện từ các demo đã ghi \[4, 5\]. Teleop cho điều khiển tức thì, dễ hiểu nhưng thường cần phần cứng đắt như exoskeleton, cặp tay dẫn–tay theo, hoặc kính VR \[7, 8\]. Học bắt chước giảm gánh nặng demo nhưng đòi hỏi thu dữ liệu cẩn thận, huấn luyện bằng GPU và đánh giá chính sách.

Các cách teleop không marker gần đây như AnyTeleop \[13\] và Dex-Cap \[14\] cho thấy theo dõi tay bằng thị giác có thể thay phần cứng chuyên dụng. Tuy nhiên chúng thường cần bộ ước lượng tư thế chạy GPU, hệ nhiều camera, hoặc bàn tay robot nhiều ngón. Chưa có công trình nào đưa ra một pipeline retargeting giải tích hoàn chỉnh, **chỉ dùng CPU**, từ một camera RGB-D góc nhìn thứ nhất tới một tay máy giá rẻ — không cần dữ liệu huấn luyện, không GPU, không phần cứng chuyên dụng ngoài phụ kiện in 3D.

Bài này nghiên cứu cách một pipeline IK giải tích như vậy nối hai hướng teleop và học bắt chước: biến bản ghi RGB-D góc nhìn thứ nhất thành quỹ đạo robot có thể phát lại trên phần cứng và dùng lại làm dữ liệu huấn luyện. Pipeline được thiết kế để **sinh quỹ đạo offline** chứ không điều khiển thời gian thực, nhờ đó xử lý quỹ đạo chất lượng cao mà không bị ràng buộc độ trễ. Đóng góp:

1. Một pipeline đầu–cuối chỉ dùng CPU, từ video RGB-D góc nhìn thứ nhất tới quỹ đạo robot một tay qua IK bình phương tối thiểu có giảm chấn, không cần dữ liệu huấn luyện.
2. Quy trình chuyển từ mô phỏng sang thực tế dùng PyBullet để xem trước và kiểm tra quỹ đạo trước khi chạy trên robot SO-ARM101 thật.
3. So sánh định lượng retargeting IK giải tích với bốn chính sách VLA trên bài gắp–đặt có cấu trúc, có khoảng tin cậy thống kê qua ba lần chạy độc lập.
4. Phân tích sai số toàn diện gồm độ chính xác bộ giải IK (sai số vị trí trung bình 36,4 mm), độ mượt quỹ đạo (lọc EMA giảm jerk 57–68%) và thí nghiệm bỏ bớt tham số EMA.
5. Đánh giá ngoài đời thực trong siêu thị và hiệu thuốc để xem độ bền trước cảnh bừa bộn và che khuất.

Qua các nghiên cứu này, pipeline IK đạt 86,7% ± 4,2% thành công trên bài kiểm tra có cấu trúc mà không cần huấn luyện, còn việc bàn tay bị che khuất nổi lên là hạn chế chính trong cảnh không cấu trúc.

**★ OpenArm v1.0:** ý "biến video người thành quỹ đạo robot để phát lại và làm dữ liệu huấn luyện" khớp với cả nhánh Thái Cực lẫn hướng imitation learning của nhóm, kể cả khi chưa làm được teleop thời gian thực.

## II. Công trình liên quan

### II-A. Ước lượng tư thế bàn tay

Theo dõi bàn tay thời gian thực tiến bộ rất nhanh. MediaPipe Hands \[2\] chạy ngay trên thiết bị ở 30 Hz với pipeline hai tầng BlazePalm + landmark dự đoán 21 điểm mốc, không cần GPU. Chúng tôi dùng MediaPipe vì nó nhẹ trên CPU và chạy được trên nhiều nền tảng. WiLoR \[1\] chính xác hơn nhờ bộ định vị DarkNet cộng bộ dựng 3D dựa trên ViT, đạt kết quả tốt nhất trên bộ FreiHAND và HO3D ở hơn 130 FPS, nhưng cần GPU. Cả hai đều xuất 21 điểm tương thích cấu trúc mô hình bàn tay MANO \[3\]. Điểm quan trọng: bộ định vị DarkNet của WiLoR chịu che khuất một phần tốt hơn BlazePalm của MediaPipe; chúng tôi đánh giá WiLoR như một cách giảm che khuất ở mục VI-L.

### II-B. Hệ thống teleop

Open-TeleVision \[7\] cho teleop dựa trên VR lập thể cho robot hình người. Bunny-VisionPro \[8\] dùng Apple Vision Pro để điều khiển khéo léo hai tay có phản hồi xúc giác. Các hệ này đạt độ trung thực cao nhưng cần phần cứng VR chuyên dụng. Cách của chúng tôi chỉ cần một camera độ sâu RGB-D và phụ kiện in 3D.

### II-C. Retargeting bàn tay không marker

AnyTeleop \[13\] cung cấp khung thống nhất để retarget tư thế bàn tay sang nhiều loại đầu công tác robot bằng tối ưu hoá tăng tốc GPU. Dex-Cap \[14\] thu demo thao tác khéo léo bằng găng tay bắt chuyển động. H2O \[15\] học khả năng tương tác tay–vật để retarget. Các cách này nhắm tới bàn tay robot nhiều ngón hoặc cần suy luận GPU. Pipeline của chúng tôi khác ở sự đơn giản: một camera RGB-D, MediaPipe chạy CPU, IK giải tích, không dữ liệu huấn luyện.

### II-D. Học bắt chước

ACT \[4\] đưa ra cách chia hành động thành khối (action chunking) bằng transformer cho thao tác hai tay tinh vi, đạt 80–90% thành công với 10 phút demo. SmolVLA \[5\] là mô hình VLA 450 triệu tham số, huấn luyện được trên một GPU. π₀ \[6\] là mô hình nền tảng robot tổng quát dùng flow matching trên xương sống VLM, huấn luyện trên 7 loại robot và 68 tác vụ.

### II-E. Robot giá rẻ

SO-ARM100/101 là tay 6 bậc tự do dùng servo bus STS3215 mô-men 30 kg·cm \[12\]. LeRobot \[11\] là khung Python không phụ thuộc phần cứng, chuẩn hoá thu dữ liệu, huấn luyện và triển khai trên nhiều nền tảng robot.

### II-F. Mô phỏng vật lý

PyBullet \[10\] cung cấp động lực học vật rắn thời gian thực với IK tích hợp qua Bullet Physics SDK. Nó hỗ trợ nạp URDF, điều khiển vị trí/vận tốc/mô-men, và kết xuất tăng tốc GPU, phù hợp cho làm mẫu nhanh và chuyển từ mô phỏng sang thực tế.

## III. Tổng quan pipeline

Toàn bộ chuỗi xử lý — từ thu RGB-D góc nhìn thứ nhất, qua phát hiện tay, chiếu ngược 3D, đổi toạ độ, giải IK, tới lệnh khớp robot — gọi là *pipeline retargeting IK* (Hình 1). Nó cần một camera Intel RealSense D400 gắn trên kính in 3D và một tay theo SO-ARM101.

*Hình 1* mô tả luồng dữ liệu qua các khâu:

1. **(A) Camera RGB-D** → ảnh RGB + bản đồ độ sâu
2. **(B) Phát hiện điểm mốc tay (MediaPipe)** → 21 điểm (uᵢ, vᵢ), i = 0…20
3. **(C) Chiếu ngược theo độ sâu** → **P**\_cam, toạ độ 3D trong khung camera
4. **(D) Đổi toạ độ** → **P**\_robot, toạ độ 3D trong khung robot
5. **Tính tư thế đích** → (**p**\_target, **q**\_target) trong khung robot
6. **(E) Bộ giải IK + bộ điều khiển kẹp** → góc khớp **q**
7. **(F) Xem trước trong mô phỏng**
8. **(G) SO-ARM101**

### III-A. Phần cứng

Nền tảng cảm biến góc nhìn thứ nhất gồm một camera độ sâu stereo Intel RealSense dòng D400 gắn trên kính in 3D từ ba chi tiết PLA/ABS (khung, hai gọng, giá gắn camera), bắt bằng vít M1.5, M2, M3. Camera nối qua USB-C 3.1 Gen 1, truyền RGB và độ sâu đồng bộ ở 640×480, 30 FPS. Robot là một tay theo SO-ARM101 có 6 khớp quay (5 khớp tay + 1 kẹp), dùng servo bus Feetech STS3215.

*Chú thích Hình 2:* Thiết lập phần cứng. Trái: ảnh có chú thích camera RealSense D435i, giá gắn nghiêng, chân đỡ camera, tay 6 bậc tự do và pin 12 V khi triển khai ngoài đời. Phải: sơ đồ cho thấy camera nghiêng θ = 50° dưới phương ngang và độ cao lệch t\_z = 0,48 m so với đế robot.

*(Ghi chú: chú thích nhắc tới "chân đỡ camera", và phép đổi toạ độ ở mục IV-D dùng một góc nghiêng và độ cao cố định so với đế robot. Vì vậy có vẻ khi đo, camera được đặt cố định cạnh robot chứ không di chuyển theo đầu người; bài không nói rõ điểm này.)*

## IV. Phương pháp

### IV-A. Thu RGB-D và mô hình camera

Camera RealSense cung cấp khung RGB-D đã căn khớp qua RealSense SDK. Điểm 3D trong khung camera ký hiệu **P**\_cam = (X, Y, Z)ᵀ; camera được mô hình hoá bằng phép chiếu lỗ kim (pinhole):

```latex
\begin{bmatrix}u\\ v\end{bmatrix}=\begin{bmatrix}f_{x}&0&c_{x}\\ 0&f_{y}&c_{y}\end{bmatrix}\begin{bmatrix}X/Z\\ Y/Z\\ 1\end{bmatrix}
```

với (f\_x, f\_y) là tiêu cự, (c\_x, c\_y) là điểm chính, tất cả được lấy tự động từ luồng camera khi khởi tạo. Camera ghi được file .bag để xử lý offline và MP4 để lưu trữ.

### IV-B. Ước lượng tư thế bàn tay

Dùng MediaPipe Hands để phát hiện tay 2D và định vị điểm mốc. Kiến trúc hai tầng — bộ phát hiện BlazePalm định vị tay bằng hộp bao có hướng, tiếp theo là mạng hồi quy điểm mốc nhẹ — dự đoán 21 điểm mỗi tay theo thời gian thực trên CPU. Đầu ra gồm nhãn tay trái/phải và toạ độ 2D {(uᵢ, vᵢ)}, i = 0…20, phủ cổ tay và các khớp ngón cái, trỏ, giữa, áp út, út.

Gọi **P**ₜ^raw là mảng điểm mốc thô của MediaPipe tại thời điểm t. Để giảm rung theo thời gian, áp lọc trung bình động mũ (EMA):

```latex
\mathbf{P}_{t}=\alpha\,\mathbf{P}_{t}^{\text{raw}}+(1-\alpha)\,\mathbf{P}_{t-1},\quad\alpha=0.8
```

với **P**ₜ là mảng đã làm mượt; toạ độ (uᵢ, vᵢ) đã làm mượt được dùng cho các khâu sau.

### IV-C. Dựng 3D theo độ sâu

Định nghĩa hai điểm mốc tham chiếu cho kẹp: **P**₁ (khớp bàn–đốt ngón cái, MCP) và **P**₂ (MCP ngón trỏ), theo cấu trúc 21 điểm của MediaPipe. Mỗi điểm 2D (uᵢ, vᵢ) được chiếu ngược thành điểm 3D **P**\_cam bằng ảnh độ sâu D (mét) và nội tham camera:

```latex
\mathbf{P}_{\text{cam}}=D[u_{i},v_{i}]\begin{bmatrix}(u_{i}-c_{x})/f_{x}\\ (v_{i}-c_{y})/f_{y}\\ 1\end{bmatrix}
```

với D\[uᵢ, vᵢ\] là độ sâu tại điểm ảnh đó. Độ sâu ngoài khoảng hợp lệ 0,1–5,0 m bị đánh dấu không hợp lệ.

*Cơ chế thay thế độ sâu:* khi đúng một trong hai điểm **P**₁, **P**₂ có độ sâu hỏng, lấy độ sâu của điểm còn lại thay vào, vì hai điểm kề nhau này thường có độ sâu gần nhau. Một tư thế tay bị loại hoàn toàn nếu dưới 50% trong 21 điểm có độ sâu hợp lệ.

### IV-D. Đổi từ khung camera sang khung robot

Cả 21 điểm trong khung camera được đổi sang khung đế robot:

```latex
\mathbf{P}_{\text{robot}}=\mathbf{R}\,\mathbf{P}_{\text{cam}}+\mathbf{t}
```

Từ đây mọi ký hiệu điểm mốc (**P**₁, **P**₂, đầu ngón…) đều là toạ độ khung robot sau phép đổi này. Ma trận quay **R** tính tới góc nghiêng camera θ = 50° dưới phương ngang (Hình 2):

```latex
\mathbf{R}=\begin{bmatrix}-1&0&0\\ 0&\sin\theta&-\cos\theta\\ 0&-\cos\theta&-\sin\theta\end{bmatrix}
```

với véc tơ tịnh tiến **t** = (0,04; −0,049; 0,48)ᵀ mét. Các tham số này — gồm góc gắn 50° và độ lệch tịnh tiến — được lấy thẳng từ bản lắp ráp CAD SolidWorks mô hình cả robot lẫn giá gắn camera ở đúng vị trí vật lý. Việc đảo dấu trục x là phản chiếu gương ảnh theo phương ngang (camera nhìn tay từ góc thứ nhất, còn robot quay mặt về phía người vận hành), còn phép quay quanh trục x ánh xạ hướng nhìn xuống của camera về hệ trước–lên của robot.

### IV-E. Tính tư thế đích

Tư thế đích của đầu công tác (**p**\_target, **q**\_target) gồm vị trí và quaternion hướng, cả hai trong khung đế robot. Vị trí đích là trung điểm hai điểm mốc kẹp:

```latex
\mathbf{p}_{\text{target}}=\tfrac{1}{2}\bigl(\mathbf{P}_{1}+\mathbf{P}_{2}\bigr)
```

Hướng đích **q**\_target lấy từ ba trục trực giao dựng từ hình học bàn tay (Hình 3). Trục thứ nhất là hướng bề rộng kẹp:

```latex
\mathbf{e}_{1}=\frac{\mathbf{P}_{2}-\mathbf{P}_{1}}{\|\mathbf{P}_{2}-\mathbf{P}_{1}\|}
```

Hướng chỉ trung bình của ngón tay **d** tính từ hai vector đơn vị **u**\_thumb và **u**\_index, mỗi vector đi từ MCP tới đầu ngón tương ứng:

```latex
\mathbf{d}=\frac{\mathbf{u}_{\text{thumb}}+\mathbf{u}_{\text{index}}}{\|\mathbf{u}_{\text{thumb}}+\mathbf{u}_{\text{index}}\|}
```

Lấy trung bình các vector **đơn vị** thay vì vector hướng ngón thô để hai ngón đóng góp ngang nhau vào hướng chỉ, bất kể chiều dài 3D đo được khác nhau do nhiễu độ sâu. Hai trục còn lại lấy bằng tích có hướng:

```latex
\mathbf{e}_{3}=\frac{\mathbf{e}_{1}\times\mathbf{d}}{\|\mathbf{e}_{1}\times\mathbf{d}\|},\qquad \mathbf{e}_{2}=\mathbf{e}_{3}\times\mathbf{e}_{1}
```

Quaternion **q**\_target là quaternion tạo từ ma trận \[**e**₁ **e**₂ **e**₃\]. Khi thiếu điểm đầu ngón (bị che hoặc hỏng độ sâu), dùng *hướng dự phòng* chỉ từ **P**₁, **P**₂ và cổ tay, thay **d** bằng vector từ cổ tay tới tâm kẹp.

*Chú thích Hình 3:* Các vector hướng kẹp. **e**₁ là hướng bề rộng kẹp đã chuẩn hoá (**P**₁ → **P**₂). **u**\_thumb và **u**\_index là vector đơn vị từ mỗi MCP tới đầu ngón; trung bình chuẩn hoá **d** của chúng (Pt. 8) làm hướng chỉ tham chiếu. **d** nhìn chung **không** vuông góc với **e**₁; khung trực chuẩn có được nhờ tính **e**₃ = **e**₁ × **d** (chuẩn hoá, hướng ra ngoài trang) và **e**₂ = **e**₃ × **e**₁ (Pt. 9–10).

**★ OpenArm v1.0:** cách dựng khung kẹp từ MCP ngón cái, MCP ngón trỏ và hai đầu ngón (mục IV-E) dùng thẳng được cho hướng cổ tay J5–J7 và kẹp của OpenArm. Phần độ sâu (IV-C) cần camera RGB-D; với webcam thường, nhóm phải thay bằng điểm 3D thế giới của MediaPipe hoặc hai camera.

### IV-F. Động học ngược

Với tư thế đích (**p**\_target, **q**\_target), góc khớp được tìm bằng bài toán tối ưu bình phương tối thiểu có giảm chấn:

```latex
\mathbf{q}^{*}=\arg\min_{\mathbf{q}}\;\|\text{FK}(\mathbf{q})-(\mathbf{p}_{\text{target}},\mathbf{q}_{\text{target}})\|^{2}+\lambda\,\|\mathbf{q}-\mathbf{q}_{\text{rest}}\|^{2}
```

Bộ giải dùng engine IK của PyBullet, tối đa **100 vòng lặp**, ngưỡng phần dư **10⁻⁴**. FK(·) ánh xạ góc khớp sang tư thế đầu công tác; **q**\_rest là trạng thái khớp hiện tại, dùng làm tư thế nghỉ; λ là hệ số giảm chấn theo từng khớp lấy từ URDF.

5 khớp quay của tay không thể tự thỏa mãn một tư thế đích 6 bậc tự do đầy đủ. Cách phát biểu DLS ưu tiên độ chính xác vị trí hơn hướng.

Một bộ lọc EMA thứ hai (α\_IK = 0,5) giảm rung trong không gian khớp:

```latex
\hat{\mathbf{q}}_{t}=\alpha_{IK}\,\mathbf{q}^{*}_{t}+(1-\alpha_{IK})\,\hat{\mathbf{q}}_{t-1}
```

Bộ lọc này khác bộ EMA ở không gian điểm mốc (α\_lm = 0,8) vốn làm mượt điểm 2D trước khi chiếu ngược. Kiểm tra an toàn loại bỏ mọi đích có **z < 0,05 m** để tránh đâm xuống mặt sàn.

### IV-G. Điều khiển kẹp

Góc kẹp đích φ\_target tính từ góc giữa hai vector đi từ tâm kẹp (**p**\_target) tới đầu ngón cái và đầu ngón trỏ:

```latex
\varphi_{\text{target}}=\arccos\!\left[\frac{(\mathbf{p}_{\text{thumb}}-\mathbf{p}_{\text{target}})\cdot(\mathbf{p}_{\text{index}}-\mathbf{p}_{\text{target}})}{\|\mathbf{p}_{\text{thumb}}-\mathbf{p}_{\text{target}}\|\;\|\mathbf{p}_{\text{index}}-\mathbf{p}_{\text{target}}\|}\right]
```

Giá trị được kẹp về \[0, π/2\], cộng offset −0,175 rad để kẹp chặt hơn, rồi kẹp vào giới hạn khớp kẹp \[φ\_min, φ\_max\] = \[0,087; 1,658\] rad.

Cơ chế dự phòng nhiều mức khi thiếu đầu ngón: (1) dùng điểm khớp ngón tương ứng; (2) nếu vẫn hỏng, giữ góc kẹp hợp lệ gần nhất; (3) cuối cùng đặt kẹp ở mức mở giữa (φ\_min + φ\_max)/2. Nhờ vậy mất một điểm mốc không gây hành vi kẹp thảm hoạ.

### IV-H. Xem trước trong mô phỏng

Trước khi chạy thật, mọi quỹ đạo từ IK đều được xem trước trong PyBullet. Mô phỏng nạp URDF SO-ARM101 với 7 khớp mỗi tay (1 đế cố định + 5 khớp quay tay + 1 khớp quay kẹp); thí nghiệm chỉ dùng tay phải. Mô phỏng chạy ở **240 Hz**. Tham số PID điều khiển vị trí được chỉnh theo thực nghiệm cho khớp tốc độ và đáp ứng của robot thật.

*Chú thích Hình 4:* Xem trước trong PyBullet. Trên trái: khung RGB từ camera góc thứ nhất thấy tay người vận hành. Trên phải: bản đồ màu độ sâu. Dưới: tay robot trong PyBullet với nhãn khớp và điểm đích IK (cầu xanh/đỏ), bám theo quỹ đạo lấy từ tay.

Giao diện gồm ba khung: ảnh RGB góc thứ nhất, bản đồ độ sâu, và robot đang bám đích IK — để kiểm tra quỹ đạo khớp tái hiện đúng chuyển động tay trước khi chạy thật. Mô phỏng cũng xuất dữ liệu demo (file hành động .npy) để huấn luyện chính sách học bắt chước.

### IV-I. Triển khai trên robot thật

Hành động đã kiểm tra được chạy trên SO-ARM101 thật qua LeRobot. Mỗi góc khớp qᵢ được ánh xạ tuyến tính sang hành động chuẩn hoá:

```latex
a_{\text{norm}}=\frac{q_{i}-q_{\min}}{q_{\max}-q_{\min}}\in[0,1]
```

rồi đổi sang lệnh motor theo quy ước LeRobot cho servo STS3215:

```latex
a_{\text{motor}}=\begin{cases}(a_{\text{norm}}-0.5)\times 200 & \text{khớp tay}, \in[-100,100]\\ a_{\text{norm}}\times 100 & \text{kẹp}, \in[0,100]\end{cases}
```

Hệ số PID servo được cấu hình cho chuyển động mượt (Bảng II; nội dung Bảng I giới hạn khớp và Bảng II PID mình không lấy được từ trang gốc), có giới hạn gia tốc và vận tốc để tránh chuyển động đột ngột. Có ba chế độ kẹp: Normal (truyền thẳng góc), Binary (ngưỡng 60°, chỉ mở hẳn hoặc đóng hẳn), Offset (cộng thêm độ lệch cấu hình được để kẹp chặt hơn).

*Chú thích Hình 5:* Bắt chước tay. Trái: người vận hành đeo kính RealSense thực hiện thao tác gắp. Phải: robot SO-ARM101 lặp lại tư thế tay qua pipeline IK.

**★ OpenArm v1.0:**

- Hàm mục tiêu IV-F (bám tư thế + phạt lệch khỏi tư thế hiện tại) là mẫu tốt cho IK trên URDF v1.0. OpenArm 7 khớp nên dư một bậc thay vì thiếu một bậc như SO-101; nên thêm một số hạng kéo khuỷu về phía khuỷu người.
- Hai tầng lọc EMA (điểm mốc 0,8 và góc khớp 0,5) và kẹp có 3 mức dự phòng là thứ có thể làm ngay.
- Kiểm tra "z < 0,05 m thì bỏ" tương ứng với việc OpenArm cần giới hạn vùng làm việc (không đập vào bàn, không va thân).
- OpenArm dùng motor Damiao qua CAN, không phải servo Feetech; phần chuẩn hoá \[−100, 100\] ở IV-I không áp dụng, thay bằng góc theo rad/độ của `openarm_can` hoặc `openarm_follower`.

## V trở đi: chưa dịch

Công cụ đọc web của mình chỉ lấy được văn bản bài tới đầu mục V-A. Các lần đọc phần sau trả về số liệu không có trong trang gốc, nên mình không đưa vào. Các mục còn thiếu: V (thiết lập thí nghiệm, huấn luyện VLA), VI-A–VI-M (độ trễ, tỉ lệ thành công, so sánh với ACT/SmolVLA/π₀.₅/GR00T, thử ngoài đời, độ mượt, WiLoR, các kiểu thất bại), VII (kết luận).

Số liệu đã kiểm chứng từ phần tóm tắt: 86,7% ± 4,2% thành công trên lưới 5 ô (10 lần gắp mỗi ô, 3 lần chạy); sai số vị trí IK trung bình 36,4 mm; EMA giảm jerk 57–68%; ngoài đời còn 9,3%; WiLoR tăng tỉ lệ phát hiện tay thêm 8%.
