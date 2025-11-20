FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ----------------------------------------------------------------------
# 1. 기본 패키지
# ----------------------------------------------------------------------
RUN apt update && apt install -y \
    lsb-release curl gnupg2 build-essential \
    python3 python3-pip python3-colcon-common-extensions \
    git nano sudo wget iputils-ping \
    libgl1-mesa-glx libglib2.0-0

# ----------------------------------------------------------------------
# 2. ROS2 Humble 설치
# ----------------------------------------------------------------------
RUN curl -sSL https://raw.githubusercontent.com/ros2/ros2/master/ros2.repos -o ros2.repos

RUN apt update && apt install -y \
    locales && locale-gen en_US en_US.UTF-8
ENV LANG en_US.UTF-8

RUN apt update && apt install -y \
    software-properties-common && \
    add-apt-repository universe

RUN curl -sSL https://raw.githubusercontent.com/ros2/ros2/master/ros.key \
    | sudo apt-key add -

RUN apt update && apt install -y \
    ros-humble-desktop \
    ros-humble-cv-bridge \
    ros-humble-vision-msgs \
    ros-humble-vision-opencv \
    ros-humble-tf2-tools \
    ros-humble-camera-info-manager \
    ros-humble-nav2* \
    ros-humble-moveit*

# ----------------------------------------------------------------------
# 3. YOLO + OpenCV + AI 관련
# ----------------------------------------------------------------------
RUN pip install --upgrade pip
RUN pip install ultralytics onnxruntime-gpu filterpy pyzbar scipy numpy transformers whisper

# ----------------------------------------------------------------------
# 4. RealSense(optional)
# ----------------------------------------------------------------------
RUN apt install -y ros-humble-realsense2*

# ----------------------------------------------------------------------
# 5. Doosan ROS2 Driver
# ----------------------------------------------------------------------
RUN git clone https://github.com/doosan-robotics/doosan-robot2.git /opt/dsr \
    && cd /opt/dsr \
    && rm -rf .git

# ----------------------------------------------------------------------
# 6. 워크스페이스 생성
# ----------------------------------------------------------------------
RUN mkdir -p /ros2_ws/src
WORKDIR /ros2_ws

COPY ./src ./src

RUN /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build --symlink-install"

# ----------------------------------------------------------------------
# 7. ENTRYPOINT (산업용 신뢰성 우선)
# ----------------------------------------------------------------------
COPY docker_entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "bringup_pkg", "full_bringup.launch.py"]
