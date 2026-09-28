#!/usr/bin/env bash
# Tải model MediaPipe (Pose full + Hand) vào thư mục models/
set -e
cd "$(dirname "$0")/.."
mkdir -p models
BASE=https://storage.googleapis.com/mediapipe-models
curl -L -o models/pose_landmarker_full.task  $BASE/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
curl -L -o models/pose_landmarker_lite.task  $BASE/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
curl -L -o models/hand_landmarker.task       $BASE/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
ls -lh models
