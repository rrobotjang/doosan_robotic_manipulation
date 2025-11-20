#!/bin/bash
set -e

# 실시간 스케줄링
ulimit -r unlimited || true

# ROS 초기화
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash

# GPU 초기화 로그
nvidia-smi || echo "NVIDIA GPU not found!"

exec "$@"
