from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Camera
        Node(package="vision_perception", executable="camera_driver_node"),
        Node(package="vision_perception", executable="yolo_detector"),

        # HRI Hand Tracking
        Node(package="hri_gesture", executable="hand_tracking"),

        # Object Tracking
        Node(package="vision_perception", executable="object_tracker"),

        # Manipulation
        Node(package="manipulation", executable="pick_place_node"),

        # Doosan Driver
        Node(package="doosan_bringup3", executable="dsr_driver"),
        # 전체 Integration Launch File
        # bringup.launch.py
        Node(
            package="vision_perception",
            executable="yolo_bytetrack_node",
            output="screen"
        ),

        Node(
            package="manipulation",
            executable="pickplace_dsr_node",
            output="screen"
        ),
    ])
