# Bring-up OpenArm v1.0 (làm trước khi chạy teleop)
1. `./setup_can.sh` — bật can0/can1 (1 Mbps / 5 Mbps CAN-FD) cho SavvyCAN-FD-X2 (hiện ra là PEAK, driver peak_usb).
2. `python3 read_joints.py --iface can0` — chỉ đọc góc, không mô-men. Xác định can0/can1 là tay nào.
3. `python3 wiggle_j7.py --iface can0` — lắc J7 ±8°, tay thả xuôi, có E-stop.
Không chạy `set_zero` trừ khi cả nhóm quyết định hiệu chuẩn lại (v1.0: `--robot-version v1`).
