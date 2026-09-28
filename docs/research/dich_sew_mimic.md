# Bản dịch: SEW-Mimic

Bài gốc: *A Closed-Form Geometric Retargeting Solver for Upper Body Humanoid Robot Teleoperation* (Bộ giải retargeting hình học dạng đóng cho teleop phần thân trên robot hình người), [arXiv 2602.01632](https://arxiv.org/html/2602.01632). Tác giả: Chuizheng Kong, Yunho Cho, Wonsuhk Jung, Idris Wibowo, Parth Shinde, Sundhar Vinodh-Sangeetha, Long Kiu Chung, Zhenyang Chen, Andrew Mattei, Advaith Nidumukkala, Alexander Elias, Danfei Xu, Taylor Higgins, Shreyas Kousik (Georgia Tech, Qualcomm, Standard Bots, FAMU-FSU).

Dịch theo đúng thứ tự các mục của bản HTML trên arXiv. Hình vẽ không chép lại, chỉ dịch chú thích. Công cụ đọc web của mình trả về văn bản bài báo, đôi chỗ đã rút gọn nhẹ câu chữ; các con số, tên thuật toán và công thức được giữ nguyên. Chỗ nào cần trích dẫn chính xác, bạn nên đối chiếu bản gốc.

## Tóm tắt (Abstract)

Ánh xạ chuyển động người sang tư thế robot (retargeting) là cách thực tế để teleop hai tay của robot hình người. Nhưng các phương pháp hiện có có thể không tối ưu và chậm, thường gây chuyển động không mong muốn hoặc độ trễ. Bài báo đặt lại vấn đề thành **bài toán căn chỉnh hướng (orientation alignment)**, từ đó có được một thuật toán hình học dạng đóng (closed-form) có bảo đảm tối ưu.

Ý chính là căn tay robot theo hướng cánh tay trên và cẳng tay của người, lấy từ các điểm mốc vai (Shoulder), khuỷu (Elbow), cổ tay (Wrist) — vì vậy phương pháp có tên **SEW-Mimic**. Ưu điểm chính là suy luận nhanh (3 kHz) trên CPU thương mại thông thường, còn dư thời gian cho các ứng dụng phía sau như bộ lọc an toàn tránh hai tay tự va chạm.

Phương pháp dùng được cho hầu hết tay robot và robot hình người 7 bậc tự do, không phụ thuộc nguồn điểm mốc. Thí nghiệm cho thấy nó tốt hơn các phương pháp retargeting khác cả về tốc độ tính và độ chính xác. Nghiên cứu người dùng thử nghiệm cho thấy khả năng tăng tỉ lệ hoàn thành nhiệm vụ teleop. Phân tích sơ bộ cho thấy dữ liệu thu bằng SEW-Mimic mượt hơn, có lợi cho học chính sách (policy learning). Phương pháp cũng dùng được để tăng tốc retargeting toàn thân cho robot hình người.

## I. Giới thiệu

*Chú thích Hình 1:* Chúng tôi đề xuất SEW-Mimic để ánh xạ giải tích các điểm mốc vai, khuỷu, cổ tay (SEW) của người sang góc khớp tay robot. Ý tưởng then chốt là dùng **sai số hướng của các đoạn chi và của bàn tay/dụng cụ** làm thước đo độ giống tư thế, vì nó cho phép có lời giải dạng đóng, tối ưu có chứng minh và **không phụ thuộc tỉ lệ kích thước**, như minh hoạ trên Kinova Gen3, Rainbow RB-Y1 và Unitree G1 (vẽ đúng tỉ lệ). Vì SEW-Mimic tính rất nhanh, chúng tôi dùng nó làm bộ giải IK bên trong một bộ lọc an toàn chống tự va chạm khi teleop hai tay.

Retargeting là quá trình ánh xạ tư thế người sang cấu hình robot. Trong teleop, nó cho con người cách điều khiển tự nhiên một robot nhiều bậc tự do, đồng thời thu được dữ liệu huấn luyện cho chính sách tự hành. Tuy vậy, khác biệt vật lý giữa người và robot — kích thước, số bậc tự do, ràng buộc chuyển động — khiến bài toán không đơn giản.

Vì thế, phần lớn hệ thống hiện nay chỉ bám theo chuyển động bàn tay người. Họ coi retargeting là bài toán **IK cho đầu công tác (end-effector)** dùng ma trận Jacobian của robot, ánh xạ vận tốc 6 bậc tự do của đầu công tác sang vận tốc khớp. Cách này không teleop tốt được khi robot ở gần giới hạn chuyển động hoặc gần tư thế kỳ dị, nơi phép nghịch đảo bị suy biến. Kỹ thuật giả nghịch đảo (pseudo-inverse) hoặc IK tối ưu hoá có thể giảm suy biến cho tay 6 bậc tự do, nhưng tay robot hình người thường có 7 bậc tự do để giống người. Khi chỉ retarget bàn tay 6 bậc, bậc thứ 7 — tương ứng khuỷu tay người — sinh ra **chuyển động trong không gian rỗng (null-space) không mong muốn**, làm tăng nguy cơ va chạm.

Để khắc phục, các cách tiếp cận humanoid dựa trên học gần đây tối ưu trên góc khớp robot để khớp với các điểm mốc cơ thể người (vai, khuỷu, cổ tay, cổ chân). Đưa khuỷu vào bằng cách tối thiểu khoảng cách Euclid giải quyết được lỗ hổng trên, nhưng giải bài toán tối ưu đó thường mất khoảng **0,7 giây**. Trong teleop thời gian thực để thu dữ liệu, độ trễ như vậy tạo ra hành vi ngập ngừng, làm giảm chất lượng chính sách được huấn luyện.

Bài báo giải quyết cả hai hạn chế — chỉ bám bàn tay và độ trễ suy luận — bằng SEW-Mimic, một thuật toán retargeting hình học (dựa trên điểm mốc), dạng đóng, tối ưu có chứng minh. SEW-Mimic không dùng Jacobian và không tối ưu lặp. Nó phân rã hình học: định nghĩa vector chi của người là vector đơn vị giữa các điểm mốc, rồi giải góc khớp robot sao cho các vector chi tương ứng của robot thẳng hàng với chúng, tức là làm tư thế giống nhất. Điểm then chốt: độ giống được đo bằng **sai số hướng thay vì sai số Euclid**, nên phương pháp **không cần hiệu chuẩn** giữa người và robot có kích thước khác nhau.

Đóng góp:

- Đề xuất SEW-Mimic, thuật toán nhanh và tối ưu cho retargeting và teleop phần thân trên robot hình người.
- Xây dựng bộ lọc an toàn dùng SEW-Mimic để tránh tự va chạm khi teleop hai tay.
- Cung cấp ứng dụng độc lập, mã nguồn mở, tích hợp SEW-Mimic với **MediaPipe** hoặc kính **Meta Quest** để teleop, kèm bản demo web cho thấy tốc độ tính.

Thí nghiệm cho thấy SEW-Mimic đạt độ giống tư thế cao với thời gian tính thấp, và có tác động tới tỉ lệ thành công teleop, học chính sách tự hành và tốc độ retargeting toàn thân. Các demo phần cứng trên nhiều nền tảng cho thấy tính ứng dụng thực tế.

**★ OpenArm v1.0:** lập luận về "bậc thứ 7 gây chuyển động null-space" đúng với OpenArm (7 khớp). Nếu nhóm chỉ điều khiển theo vị trí cổ tay (như repo 1, repo 4), khuỷu robot sẽ tự do và có thể va vào thân hoặc tay kia.

## II. Công trình liên quan

Các hệ teleop hai tay và toàn thân hiện đại chia thành hai nhóm: điều khiển bằng phần cứng kiểu "con rối" (mục II-A) và retargeting tư thế bằng phần mềm (mục II-B). Chúng tôi xem xét các công trình phổ biến theo hai tiêu chí: *tốc độ tính* và *độ giống tư thế* (mức thẳng hàng về hướng giữa các khâu của người và robot, như mục I). Phương pháp của chúng tôi dùng IK hình học giải tích, được bàn cuối cùng (mục II-C).

### A. Teleop dựa trên phần cứng

Cách này ghép tay robot với một tay dẫn (leader arm) để người vận hành cung cấp trực tiếp góc khớp cho robot (ví dụ ALOHA \[12\] và GELLO \[41\]). Tốc độ tính cao nhưng độ giống tư thế thấp, vì người điều khiển thẳng các khớp robot chứ robot không bắt chước dáng tay người. Có thể khắc phục bằng tay dẫn lấy con người làm trung tâm, hoạt động như buồng lái khung xương ngoài (exoskeleton) \[16, 42, 45, 4\], nhưng phần cứng đó khó tiếp cận và khó dùng cho nhiều loại robot. Chúng tôi đưa ra cách làm bằng phần mềm, nhắm tới độ giống tư thế và tốc độ tính tương đương buồng lái exoskeleton.

### B. Teleop bằng retargeting phần mềm

Các phương pháp phần mềm hỗ trợ nhiều loại robot hơn, nhưng thường thiếu tốc độ tính và/hoặc độ giống tư thế. Chúng thường lấy chuyển động người từ camera hoặc thiết bị đeo, rồi dùng IK để retarget sang chuyển động đầu công tác hoặc cả cánh tay.

**1) Retargeting đầu công tác.** Nhóm này dùng dữ liệu tư thế 6D của bàn tay người (ví dụ DexMimicGen \[18\], OpenTeach \[17\]) và giải IK ra cấu hình khớp robot bằng các bộ giải như Mink \[43\] và OSC \[19\]; tức là coi teleop như di chuyển tự do hai đầu công tác trong không gian 3D. Cách này gặp khó khi chuyển động đầu công tác bị giới hạn bởi cấu hình tay, đặc biệt ở tư thế kỳ dị khi tay duỗi thẳng, làm giảm ổn định số và tốc độ \[28\]. Ngược lại, phương pháp của chúng tôi không cần Jacobian nên không gặp vấn đề ổn định số gần điểm kỳ dị. Một số phương pháp IK tối ưu hoá khác tránh đi vào tư thế kỳ dị ngay từ đầu (ví dụ BunnyVisionPro \[8\] và Open-TeleVision \[6\] trong `xr_teleoperate` của Unitree \[34\]) bằng giả nghịch đảo Jacobian có trọng số (thường giải bằng Pinocchio \[5\]). Tuy nhiên với tay robot hình người 7 bậc tự do, các phương pháp này **không có tính tuần hoàn (non-cyclicity)** do bậc tự do dư \[9\]: khi đầu công tác quay lại một tư thế đã đi qua, khuỷu có thể không trở về vị trí cũ. Để giảm hiện tượng này, OSC \[19\] cho phép đặt mục tiêu phụ trong không gian khớp (ví dụ giữ góc khớp ban đầu) bằng cách chiếu sai số khớp vào không gian rỗng của Jacobian. Nhưng vì không thể ánh xạ thẳng tư thế khuỷu người sang không gian khớp robot mà không giải IK trước, cách này vẫn không cho người vận hành điều khiển trực tiếp khuỷu robot, khiến độ giống tư thế càng kém. Phương pháp của chúng tôi cho phép điều khiển khuỷu trực tiếp.

**2) Retargeting dựa trên điểm mốc.** Để tránh kỳ dị Jacobian và xử lý động lực học toàn thân phức tạp của robot hình người, một số phương pháp dùng hàm chi phí tối ưu hoá tuỳ biến cùng bộ dữ liệu chuyển động người lớn (ví dụ AMASS \[24\]) để huấn luyện chính sách retargeting và giữ thăng bằng thời gian thực (ví dụ H2O \[15\], CLONE \[21\], TWIST \[44\]). Độ giống tư thế cao, nhưng sai số vị trí khớp thường vẫn lớn, khiến tác vụ khéo léo khó hơn so với phương pháp lặp \[18\]. Ngoài ra độ trễ cao (khoảng 0,7 giây), một phần do tối ưu hoá hội tụ chậm cộng với truyền không dây; điều này có thể khiến người teleop dạy robot những hành vi ngập ngừng \[44\]. Chúng tôi không xử lý độ trễ truyền thông, nhưng tốc độ tính cao của phương pháp giúp giảm độ trễ tổng. SEW-Mimic có thể thay thế trực tiếp General Motion Retargeting (GMR), thứ mà TWIST dùng làm điểm khởi đầu động học. Chúng tôi coi hông–gối–cổ chân của mỗi chân như vai–khuỷu–cổ tay, cho TWIST độ chính xác retargeting tương đương GMR nhưng nhanh hơn **10 đến 1000 lần** (xem mục VI-E).

### C. Bộ giải IK hình học giải tích

Để tránh các khó khăn trên (hạn chế phần cứng, kỳ dị Jacobian, tốc độ thấp), có thể dùng bộ giải IK giải tích, vốn thường dành cho tay máy 6 bậc tự do. Hướng này có lịch sử lâu \[31, 30\], gần đây được cải tiến bởi ik-geo \[10\] và EAIK \[28\]. Đặc biệt, \[10\] đưa ra lời giải tiện lợi cho các **bài toán con Paden-Kahan** \[30\] — nền tảng của IK giải tích — qua phân rã hình học dạng đóng. Một công trình tiếp nối ik-geo là stereo-sew \[9\], tham số hoá khớp dư của tay 7 bậc bằng góc khuỷu và dùng phân rã bài toán con cho IK tư thế đầu công tác; kết quả là lời giải dạng đóng hoặc tìm kiếm ít chiều, tuỳ hình thái robot. Các cách này tính rất nhanh nhưng chỉ là IK thuần tuý, không nhắm tới độ giống tư thế. SEW-Mimic nhắm trực tiếp tới độ giống tư thế mà vẫn giữ tốc độ cao, vì luôn có lời giải dạng đóng. Việc dùng góc khuỷu cho IK giải tích không mới \[20, 2, 9, 36\], nhưng theo hiểu biết của chúng tôi, cách dùng điểm mốc SEW cho retargeting là mới.

**★ OpenArm v1.0:** hiện tượng "không tuần hoàn" sẽ xảy ra nếu nhóm dùng IK Pinocchio chỉ theo cổ tay: đưa tay đi rồi về chỗ cũ, khuỷu robot có thể ở vị trí khác. Đây là lý do nên đưa khuỷu người vào bài toán.

## III. Kiến thức nền

### A. Ký hiệu

**Quy ước.** ℝ là tập số thực, ℕ là tập số tự nhiên, SO(3) là không gian ma trận quay 3 chiều. Vô hướng viết nghiêng (x ∈ ℝ); vector và ma trận viết đậm (**x** ∈ ℝⁿ, **A** ∈ ℝⁿˣᵐ). Ma trận đơn vị n chiều là **I**ₙ; mảng 0 kích thước n×m là **0**ₙˣₘ; giả nghịch đảo của **A** là **A**†. Chỉ số bắt đầu từ 1 cho khớp với cách đánh số khớp robot. Phần tử thứ i của vector là **v**\[i\], các phần tử từ i tới j là **v**\[i:j\], phần tử (i, j) của mảng là **A**\[i, j\]. Ghép hai vector: (**a**, **b**) = \[**a**ᵀ, **b**ᵀ\]ᵀ. Chỉ số dưới là nhãn, chỉ số trên là khung toạ độ.

**Phép toán thường dùng.** Với trục quay **h** ∈ ℝ³ và góc quay α ∈ ℝ, công thức Rodrigues cho ma trận quay tương ứng:

```latex
\mathscr{R}(\mathbf{h},\alpha) = \mathbf{I}_3 + (\sin\alpha)\,\operatorname{sk}(\mathbf{h}) + (1-\cos\alpha)\,(\operatorname{sk}(\mathbf{h}))^2 \qquad (1)
```

trong đó sk: ℝ³ → ℝ³ˣ³ là toán tử "mũ" (hat) trả về ma trận phản đối xứng. Chuẩn hoá vector về độ dài 1: unit(**v**) = **v** / ‖**v**‖₂. Trong thuật toán, x ← y nghĩa là gán giá trị y cho biến x.

**Khung toạ độ.** Khung gốc (baselink) của robot là khung thứ 0. Mỗi khung được biểu diễn bằng ma trận quay và vector tịnh tiến so với khung 0. Khung f là (**R**^(0,f), **p**^(0,f)), và vector **v** biểu diễn trong khung đó là **v**ᶠ. Với đối tượng trong khung quán tính, nhãn khung được bỏ khi ngữ cảnh rõ. Đổi từ khung a sang khung b:

```latex
\mathbf{v}^{b} = [\mathbf{R}^{0,b}]^{\top} (\mathbf{R}^{0,a} \mathbf{v}^{a} - \mathbf{p}^{0,a}) + \mathbf{p}^{0,b} \qquad (2)
```

Khung đánh số (thường là i) chỉ các khâu robot; khung khác dùng chữ thường (ví dụ "hm" cho người, "in" cho luồng dữ liệu điểm mốc 3D đầu vào).

### B. Mô tả tay người và tay robot

Phương pháp retarget tư thế chi của người sang chi tương ứng trên robot hình người. Để dễ trình bày, bài chủ yếu xét một chi (tay phải), minh hoạ ở Hình 2.

*Chú thích Hình 2:* So sánh tay người với các điểm mốc và tay robot phần thân trên 7 bậc tự do, gồm các khâu (hộp) và khớp (hình trụ thể hiện trục quay).

**1) Tay người.** Đầu vào là bộ điểm mốc 3D gồm vai **s**, khuỷu **e**, cổ tay **w**, cộng ma trận quay **H** biểu diễn hướng bàn tay so với khung 0. Như vậy đầu vào là (**s**, **e**, **w**, **H**) ∈ ℝ³ × ℝ³ × ℝ³ × SO(3).

*Nhận xét 1 (Khung lấy thân làm tâm):* Ký hiệu trên ngầm hiểu mọi điểm mốc nằm trong cùng một khung 0 gắn với thân; từ đây chúng tôi giả định như vậy. Phụ lục B trình bày cách đảm bảo điều này.

**2) Tay robot.** Xét chuỗi động học nối tiếp 7 bậc tự do gồm các khớp quay, trong đó **hai khớp liên tiếp quay vuông góc với nhau** (Hình 2). Khớp thứ i có góc quay qᵢ và trục quay **h**ᵢⁱ trong khung cục bộ của khâu đó (i = 1 là khớp đầu tiên); vector tư thế là **q** = \[q₁, q₂, …, qₙ\]ᵀ. Mỗi khớp i gắn với một khâu cứng có khung cục bộ biểu diễn trong khung trước đó i−1 bằng (**R**\_local^(i−1,i), **p**\_local^(i−1,i)). Chênh lệch hướng giữa khung i−1 và i:

```latex
\mathbf{R}^{i-1,i}(q_i) = \mathbf{R}^{i-1,i}_{\text{local}}\, \mathscr{R}(\mathbf{h}_i, q_i)
```

Chênh lệch hướng giữa khung i và j suy ra bằng tích các ma trận quay theo động học vật rắn thông thường. Trục **h**ᵢ biểu diễn trong khung j: **h**ᵢʲ = **R**^(j,i)(**q**) **h**ᵢⁱ. Hướng của đầu công tác robot:

```latex
\mathbf{T}(\mathbf{q}) = \prod_{i=1}^{\#\text{DOFs}} \mathbf{R}^{i-1,i}(\mathbf{q})\, \mathbf{R}_{\text{align}}
```

trong đó **R**\_align là phép biến đổi cố định để đầu công tác theo quy tắc tay phải; với bàn tay khéo léo, X/Y/Z tương ứng ngón trỏ duỗi / pháp tuyến lòng bàn tay / ngón cái duỗi, trong đó X là hướng mũi đầu công tác.

### C. Các bài toán con hình học chuẩn

Retargeting được giải bằng các bài toán con hình học có lời giải dạng đóng của Elias và Wen \[10\], mở rộng từ các phương pháp kinh điển \[30, 27\].

**Bài toán con 1 (SP1).** Cho hai vector **p**₁, **p**₂ ∈ ℝ³ và vector đơn vị **k** ∈ ℝ³ là trục quay. Quay **p**₁ quanh **k** để nó thẳng hàng với **p**₂, tức tìm góc tối ưu θ\* làm nhỏ nhất ‖ℛ(**k**, θ)**p**₁ − **p**₂‖:

```latex
\theta^{\star} \leftarrow \text{SP1}(\mathbf{p}_1, \mathbf{p}_2, \mathbf{k}) = \underset{\theta}{\arg\min}\; \| \mathscr{R}(\mathbf{k}, \theta)\, \mathbf{p}_1 - \mathbf{p}_2 \| \qquad (3)
```

SP1 được giải dạng đóng bằng Thuật toán 5 (Phụ lục A). *(Ghi chú: bản HTML viết tham số thứ ba là **u**; theo định nghĩa thì đó là trục **k**.)*

**Bài toán con 2 (SP2).** Cho hai vector **p**₁, **p**₂ ∈ ℝ³ và hai trục quay **k**₁, **k**₂ ∈ ℝ³. Làm **p**₁ thẳng hàng với **p**₂ bằng cách đồng thời quay **p**₁ quanh **k**₁ và quay **p**₂ quanh **k**₂. Tìm các cặp góc tối ưu:

```latex
\{(\theta_{1}^{\star}, \theta_{2}^{\star})_{j}\}_{j=1}^{2} \leftarrow \text{SP2}(\mathbf{p}_{1}, \mathbf{p}_{2}, \mathbf{k}_{1}, \mathbf{k}_{2}) = \underset{\theta_{1}, \theta_{2}}{\arg\min}\; \| \mathscr{R}(\mathbf{k}_{1}, \theta_{1})\, \mathbf{p}_{1} - \mathscr{R}(\mathbf{k}_{2}, \theta_{2})\, \mathbf{p}_{2} \| \qquad (4)
```

Có thể có 1 hoặc 2 nghiệm tối ưu phân biệt cho mỗi cặp góc. SP2 được giải dạng đóng bằng Thuật toán 6 (Phụ lục A), trong đó dùng Bài toán con 4 của Paden-Kahan cũng trình bày ở phụ lục.

**★ OpenArm v1.0:** điều kiện "hai khớp liên tiếp vuông góc" đúng với OpenArm: trục tại q = 0 của tay phải lần lượt là −y, −x, −z, −y, −z, +x, −y. SP1 chính là thao tác tính một góc khớp bằng `atan2`, dễ tự viết bằng numpy.

## IV trở đi: chưa dịch

Công cụ đọc web của mình chỉ lấy được văn bản bài tới hết mục III. Khi kiểm tra lại, các lần đọc mục IV–VIII và phụ lục trả về nội dung **không có trong trang gốc** (bị bịa), nên mình đã gỡ toàn bộ phần dịch đó khỏi tài liệu. Các mục còn thiếu: IV (thuật toán SEW-Mimic), V (bộ lọc an toàn tự va chạm), VI (thí nghiệm), VII–VIII (demo, kết luận, hạn chế), phụ lục A–H.

Những gì đã kiểm chứng được từ phần tóm tắt và giới thiệu: thuật toán căn **hướng** cánh tay trên và cẳng tay (từ vai, khuỷu, cổ tay) rồi tới cổ tay, giải dạng đóng bằng các bài toán con ở mục III-C, chạy khoảng 3 kHz trên CPU, và có bộ lọc an toàn chống hai tay va nhau.
