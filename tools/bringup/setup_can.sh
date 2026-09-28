#!/usr/bin/env bash
# Bật SocketCAN cho SavvyCAN-FD-X2 (nhận dạng là PEAK PCAN-USB Pro FD, driver peak_usb)
# Dùng: ./setup_can.sh            -> cấu hình can0 và can1
#       ./setup_can.sh can0       -> chỉ can0
# Thông số giống openarm-can-cli can_configure: 1 Mbps / 5 Mbps CAN-FD, sample point 0.75.
set -u
if [ $# -eq 0 ]; then IFACES=(can0 can1); else IFACES=("$@"); fi

echo "== 1. USB =="
if lsusb | grep -qi "0c72"; then
  lsusb | grep -i "0c72"
else
  echo "!! Không thấy thiết bị PEAK (0c72:xxxx). Kiểm tra cáp USB / thử cổng khác."
  lsusb
  exit 1
fi

echo "== 2. Driver =="
if ! lsmod | grep -q peak_usb; then
  echo "peak_usb chưa nạp -> modprobe"
  sudo modprobe peak_usb || { echo "!! Không nạp được peak_usb"; exit 1; }
  sleep 1
fi
lsmod | grep peak_usb

echo "== 3. Cấu hình ${IFACES[*]} =="
for i in "${IFACES[@]}"; do
  if ! ip link show "$i" &>/dev/null; then
    echo "!! Không có $i. Các interface CAN hiện có:"; ip -br link | grep -E "^can" || true
    exit 1
  fi
  if command -v openarm-can-cli &>/dev/null; then
    openarm-can-cli -i "$i" can_configure
  else
    sudo ip link set "$i" down
    sudo ip link set "$i" type can bitrate 1000000 sample-point 0.75 \
         dbitrate 5000000 dsample-point 0.75 dsjw 2 fd on restart-ms 0
    sudo ip link set "$i" up
  fi
done

echo "== 4. Trạng thái =="
for i in "${IFACES[@]}"; do
  echo "--- $i"
  ip -details link show "$i" | grep -E "state|bitrate|dbitrate|can <" | sed 's/^ */  /'
done
echo
echo "Xong. Kiểm tra có frame: candump -n 20 can0   (bật nguồn robot trước)"
