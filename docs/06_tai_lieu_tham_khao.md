# Tài liệu tham khảo

★ = đã phân tích chi tiết trong `docs/research/`.

## Bài báo chính

- ★ **SEW-Mimic** — Kong et al., 2026. *A Closed-Form Geometric Retargeting Solver for Upper Body Humanoid Robot Teleoperation*.
  [arXiv 2602.01632](https://arxiv.org/html/2602.01632). Retarget vai–khuỷu–cổ tay cho tay 7 khớp, lời giải đóng. Code phát hành sau review.
- ★ **Vision-Based Hand Shadowing** — Chiche et al., 2026. [arXiv 2603.11383](https://arxiv.org/html/2603.11383v2),
  code [chichonnade/Vision-Based-Hand-Shadowing](https://github.com/chichonnade/Vision-Based-Hand-Shadowing). MediaPipe + depth + IK DLS, SO-ARM101.
- Elias & Wen — bài toán con Paden–Kahan, IK hình học: [rpiRobotics/ik-geo](https://github.com/rpiRobotics/ik-geo) (có Python).
- Open-TeleVision — Cheng et al., CoRL 2024. [arXiv 2407.01512](https://arxiv.org/abs/2407.01512). Teleop egocentric nhập vai.
- AnyTeleop — Qin et al., RSS 2023. [arXiv 2307.04577](https://arxiv.org/abs/2307.04577) + [dex-retargeting](https://github.com/dexsuite/dex-retargeting).
- HumanPlus — [MarkFzp/humanplus](https://github.com/MarkFzp/humanplus). 1 camera RGB, WHAM + HaMeR (cần GPU), copy góc sang humanoid.
- H2O / OmniH2O — [LeCAR-Lab/human2humanoid](https://github.com/LeCAR-Lab/human2humanoid). Camera RGB → humanoid, nặng RL.

## 5 repo nhóm lọc (★ phân tích ở `research/phan_tich_5_repo.md`)

| Repo | Dùng được gì |
| --- | --- |
| ★ [ahsanali555/openarm_teleoperation](https://github.com/ahsanali555/openarm_teleoperation) | Cách dựng OpenArm với ROS 2 + MoveIt Servo (Ubuntu 22.04) |
| ★ [jasongedev/robot-teleop](https://github.com/jasongedev/robot-teleop) | Cùng ý tưởng (1 webcam → góc vai, khuỷu → 2 tay 7 khớp), nhưng code 2020 có lỗi tính góc: chỉ học ý tưởng |
| ★ [Im-ma/Teleoperation-Arm](https://github.com/Im-ma/Teleoperation-Arm) (Marionette) | Lọc từng khớp, độ tin cậy, dead-man 400 ms, soft engage 1,5 s, ghép bàn tay theo cổ tay |
| ★ [YashVG/kinetic-arm-lab](https://github.com/YashVG/kinetic-arm-lab) | Ly hợp đặt lại gốc, dừng khi mất tracking (visibility < 0,65) |
| ★ [perfanalytics/pose2sim](https://github.com/perfanalytics/pose2sim) | Ground truth nhiều camera offline (BSD-3); [Sports2D](https://github.com/davidpagnon/Sports2D) cho 1 camera |

## Repo liên quan khác

- [Nabil-Miri/mediapipe_dual_arm_control](https://github.com/Nabil-Miri/mediapipe_dual_arm_control) — 2 tay từ pose bàn tay.
- [FreeMoCap](https://github.com/freemocap/freemocap) — mocap không marker nhiều camera.
- [unitreerobotics/xr_teleoperate](https://github.com/unitreerobotics/xr_teleoperate) + televuer — pipeline XR 2 tay.
- [SpesRobotics/teleop](https://github.com/SpesRobotics/teleop) — điện thoại WebXR → pose end-effector.
- [trzy/robot-arm](https://github.com/trzy/robot-arm) — iPhone ARKit → IK → ACT.
- xlerobot-teleop — [arXiv 2603.07672](https://arxiv.org/abs/2603.07672), điện thoại trong Cardboard làm head tracking.

## OpenArm

- [enactic/openarm_description](https://github.com/enactic/openarm_description) — URDF/xacro v1.0 (commit 14ff67b dùng trong repo).
- [enactic/openarm_can](https://github.com/enactic/openarm_can) — thư viện CAN (C++ + Python), `ppa:openarm/main`.
- LeRobot OpenArm: [huggingface.co/docs/lerobot/openarm](https://huggingface.co/docs/lerobot/openarm);
  điện thoại làm tay cầm 6D: [huggingface.co/docs/lerobot/phone_teleop](https://huggingface.co/docs/lerobot/phone_teleop).

## Nhận diện tư thế

- MediaPipe Tasks (Pose Landmarker, Hand Landmarker) 0.10.21 — dùng trong repo. Bản 1.0.x đã bỏ `mp.solutions`.
- WiLoR — phát hiện + dựng tay 3D (cần GPU).

## Dữ liệu và egocentric (phương án gắn đầu)

- EgoDex — Apple, ICLR 2026, [arXiv 2505.11709](https://arxiv.org/abs/2505.11709), [apple/ml-egodex](https://github.com/apple/ml-egodex).
- HOT3D (Meta), MobileEgo Anywhere [arXiv 2605.05945](https://arxiv.org/abs/2605.05945), EgoForce [arXiv 2605.12498](https://arxiv.org/abs/2605.12498).
- EgoZero [arXiv 2505.20290](https://arxiv.org/abs/2505.20290), EgoMimic — dữ liệu kính egocentric để học policy.
- Tư thế thân từ camera đầu: Mo2Cap2 [arXiv 1803.05959](https://arxiv.org/abs/1803.05959), EgoPoser, AvatarPoser, EgoEgo.
