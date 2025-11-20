from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    pkg_perception = get_package_share_directory('perception_pkg')
    pkg_pickplace  = get_package_share_directory('pickplace_pkg')
    pkg_calib      = get_package_share_directory('calibration_pkg')
    pkg_vs         = get_package_share_directory('visual_servoing_pkg')
    pkg_hri        = get_package_share_directory('hri_pkg')
    pkg_moveit     = get_package_share_directory('dsr_moveit_config')

    ld = []

    # -----------------------
    # 1. Camera Driver
    # -----------------------
    ld.append(Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        output='screen'
    ))

    # -----------------------
    # 2. Calibration
    # -----------------------
    ld.append(TimerAction(
        period=2.0,
        actions=[Node(
            package='calibration_pkg',
            executable='wrist_cam_calibration',
            output='screen'
        )]
    ))

    # -----------------------
    # 3. Perception (YOLO + tracking)
    # -----------------------
    ld.append(Node(
        package='perception_pkg',
        executable='perception_node',
        output='screen'
    ))

    # -----------------------
    # 4. Visual Servoing
    # -----------------------
    ld.append(Node(
        package='visual_servoing_pkg',
        executable='vs_control_node',
        output='screen'
    ))

    # -----------------------
    # 5. Pick&Place Action Server
    # -----------------------
    ld.append(Node(
        package='pickplace_pkg',
        executable='pickplace_server',
        output='screen'
    ))

    # -----------------------
    # 6. HRI (Gesture + Safety + Speech)
    # -----------------------
    ld.append(Node(package='hri_pkg', executable='gesture_node'))
    ld.append(Node(package='hri_pkg', executable='speech_cmd_node'))
    ld.append(Node(package='hri_pkg', executable='safety_monitor'))

    # -----------------------
    # 7. MoveIt2
    # -----------------------
    ld.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            pkg_moveit + '/launch/move_group.launch.py'
        )
    ))

    # -----------------------
    # 8. RViz2
    # -----------------------
    ld.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            pkg_moveit + '/launch/rviz.launch.py'
        )
    ))

    return LaunchDescription(ld)
