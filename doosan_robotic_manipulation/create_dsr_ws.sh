#!/bin/bash
mkdir -p ros2_ws/src
cd ros2_ws/src

# packages
pkgs=(
  perception_pkg
  calibration_pkg
  visual_servoing_pkg
  pickplace_pkg
  hri_pkg
  bringup_pkg
)

for p in "${pkgs[@]}"; do
  ros2 pkg create $p --build-type ament_python
done

# moveit & description 패키지 자동 복사
git clone https://github.com/doosan-robotics/doosan-robot2.git dsr_description
git clone https://github.com/doosan-robotics/doosan-robot2-moveit.git dsr_moveit_config

cd ..
colcon build
