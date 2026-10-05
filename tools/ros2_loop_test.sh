#!/usr/bin/env bash
# Vòng test 100 Hz qua ROS 2 KHÔNG cần robot: chạy node teleop thật (--robot ros2) với dữ liệu thô đã ghi thay
# camera, nối với backend giả (scripts/fake_openarm_backend.py) trong container ROS 2 Humble (Python 3.10, giống máy
# backend), rồi phân tích các lệnh đã phát (scripts/analyze_ros2_log.py).
#
#   tools/ros2_loop_test.sh <thư mục dữ liệu thô> <thư mục kết quả> [tham số thêm cho shadow.py ...]
#   tools/ros2_loop_test.sh data/2026-10-03_raw_bimanual_take2 /tmp/ros2_test \
#       --replay-intrinsics /raw/intrinsics_approx.yaml \
#       --config config/d455_wrist_real.yaml --config config/gripper_real.yaml \
#       --config config/local_both_full.yaml --config config/local_3cam.yaml
#
# Trong container: repo ở /repo (chỉ đọc), dữ liệu thô ở /raw, kết quả ở /out. Đường dẫn --config tính từ /repo.
# Kết quả: /out/backend.npz (mọi lệnh backend nhận), /out/shadow.npz (--record của node), /out/ros2_lenh.png, báo cáo.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$(cd "$1" && pwd)"; OUT="$2"; shift 2
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
IMAGE="${ROS2_TEST_IMAGE:-ros:humble-ros-base}"
docker run --rm -v "$REPO":/repo:ro -v "$RAW":/raw:ro -v "$OUT":/out -e PYTHONDONTWRITEBYTECODE=1 "$IMAGE" bash -c '
set -e
echo "== cài thư viện (python $(python3 -V | cut -d" " -f2))"
apt-get update -qq && apt-get install -y -qq python3-pip libgl1 libglib2.0-0 libegl1 libgles2 >/dev/null 2>&1
pip3 install -q --retries 10 --timeout 120 "numpy<2" mediapipe==0.10.21 pyyaml matplotlib
source /opt/ros/humble/setup.bash
echo "== backend giả + node teleop"
python3 /repo/scripts/fake_openarm_backend.py --out /out/backend.npz --duration 900 --push &
BACK=$!
sleep 2
cd /repo
GLOG_minloglevel=2 python3 scripts/shadow.py --robot ros2 --source replay:/raw --headless --yes \
    --record /out/shadow.npz "$@" 2>&1 | grep -v "^W0000\|^I0000\|inference_feedback\|absl::InitializeLog\|XNNPACK"
wait $BACK
echo "== phân tích"
python3 scripts/analyze_ros2_log.py /out/backend.npz --record /out/shadow.npz --plot /out/ros2_lenh.png \
    | tee /out/bao_cao.txt
' bash "$@"
