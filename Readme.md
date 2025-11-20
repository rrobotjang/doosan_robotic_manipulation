┌────────────────────────────────────────────────────────────┐
│                       ROS2 Workspace                       │
│                            (colcon)                        │
├────────────────────────────────────────────────────────────┤
│  perception/                                               │
│     ├── camera_driver_node         (RealSense/ZED 등)      │
│     ├── depth_preprocess_node      (OpenCV)                │
│     ├── yolo_detection_node        (YOLOv8/YOLO-NAS)       │
│     ├── hand_tracking_node         (HRI 모듈)              │
│     └── object_tracking_node       (OpenCV KCF/CSRT)       │
│                                                             
│  vlm_api/                                                   │
│     └── vlm_query_node             (Vision-Language Model) │
│                                                             
│  manipulation/                                              │
│     ├── motion_planner_node        (MoveIt2)               │
│     ├── grasp_planner_node         (grasp pose)            │
│     └── pick_place_node            (Python main)           │
│                                                             
│  robot/                                                      │
│     ├── doosan_driver              (DSR ROS2)              │
│     ├── tf_broadcaster             (camera ↔ URDF)         │
│     └── bringup.launch.py                                    │
│                                                             
│  hri/                                                        │
│     └── gesture_interface_node     (손추적 이벤트)         │
│                                                             
│  docker/                                                     │
│     ├── Dockerfile.cpu                                       │
│     └── docker-compose.yaml                                  │
└────────────────────────────────────────────────────────────┘


(1) Wrist Depth Camera → YOLO + VLM 파이프라인
Depth Camera → depth_preprocess → YOLO Detection →  
      ├─ object_tracking_node (OpenCV) → Pick&Place  
      └─ hand_tracking_node (HRI) → gesture_interface

(2) Pick & Place 파이프라인
object_tracking_node → grasp_planner → motion_planner → doosan_driver

(3) HRI 손 제스처 파이프라인
hand_tracking_node (YOLO + keypoints) → gesture_interface → pick_place_node


예:

손바닥 열기 → “정지”

집게손(핀칭) → “집기 시작”

엄지 올리기 → “다음 작업 요청”


# 산업용 

ros2_ws/
├── src/
│   ├── dsr_description/        # Doosan URDF + SRDF + meshes
│   ├── dsr_moveit_config/      # MoveIt2 config
│   ├── perception_pkg/         # YOLO, Tracker, DepthFusion, Kalman
│   │    ├── launch/
│   │    │    └── perception.launch.py
│   │    ├── config/
│   │    │    └── camera.yaml
│   │    └── src/
│   │         ├── perception_node.py
│   │         ├── hand_detector.py
│   │         └── pointcloud_node.py
│   ├── calibration_pkg/
│   │    └── wrist_cam_calibration.py
│   ├── visual_servoing_pkg/
│   │    ├── launch/
│   │    └── src/
│   │         ├── vs_control_node.py
│   │         └── vs_controller.py
│   ├── pickplace_pkg/
│   │    ├── action/
│   │    │    └── PickPlace.action
│   │    ├── launch/
│   │    └── src/
│   │         ├── pickplace_server.py
│   │         └── pickplace_client.py
│   ├── hri_pkg/
│   │    ├── launch/
│   │    └── src/
│   │         ├── gesture_node.py
│   │         ├── safety_monitor.py
│   │         └── speech_cmd_node.py   # 음성 명령 → 동작 매핑
│   ├── rviz_config/
│   │    ├── rviz2_cam_pc.rviz
│   │    └── dsr_moveit.rviz
│   ├── docker/
│   │    ├── Dockerfile
│   │    └── docker-compose.yml
│   ├── bringup_pkg/
│   │    ├── launch/
│   │    │    └── full_bringup.launch.py
│   │    └── src/
│   │         └── bringup_node.py
└── colcon.meta


==================================================================================================================

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
git clone https://github.com/doosan-robotics/doosan-robot2.git dsr_description2
git clone https://github.com/doosan-robotics/doosan-robot2-moveit.git dsr_moveit_config

cd ..
colcon build


#
# 향후 계획 

신뢰성 100%용 컨테이너 최적화(Docker + GPU + ROS2)

Doosan 로봇 실사용 속도 기반 충돌 회피 모델링

AI 기반 그립 포인트 자동 최적화 (6D pose + grasp detection)

산업용 안전 규격(ISO 10218, TS 15066) 준수 구조 설계

#
#
# 전체 시스템 실행 명령
docker-compose up --build


또는 로컬에서:

ros2 launch bringup_pkg full_bringup.launch.py


4) Boston Dynamics 스타일 행동 FSM / BehaviorTree

산업용 로봇에서 가장 안정적인 구조는:

HRI → Task FSM → BehaviorTree → Skills(Pick/Place/Move/VS/Safety)

✔ BehaviorTree 구조
행동 트리 예시
Root
 └─ Sequence
      ├─ IsHumanSafe?
      ├─ DetectObject
      ├─ GenerateGraspPose
      ├─ VisualServoApproach
      ├─ ExecutePick
      ├─ MoveToPlaceLocation
      └─ ExecutePlace

✔ ROS2 BehaviorTree.CPP 노드 (C++)
bt_pkg/nodes/check_human_safe.cpp
class IsHumanSafe : public BT::ConditionNode {
public:
    IsHumanSafe(const std::string& name)
        : BT::ConditionNode(name, {}) {
        sub_ = node.create_subscription<std_msgs::msg::Bool>(
            "/safety/human_safe", 10,
            [this](auto msg){ safe_ = msg->data; });
    }

    BT::NodeStatus tick() override {
        return safe_ ? BT::NodeStatus::SUCCESS
                     : BT::NodeStatus::FAILURE;
    }

private:
    bool safe_ = true;
    rclcpp::Node node{"safe_checker"};
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr sub_;
};

✔ 고수준 FSM (Python)
fsm_pkg/task_fsm.py
from transitions import Machine

class TaskFSM(object):
    states = ["IDLE", "DETECT", "GRASP", "PICK", "PLACE", "ERROR"]

    def __init__(self):
        self.machine = Machine(model=self, states=TaskFSM.states, initial="IDLE")

        self.machine.add_transition("start", "IDLE", "DETECT")
        self.machine.add_transition("object_found", "DETECT", "GRASP")
        self.machine.add_transition("grasp_ready", "GRASP", "PICK")
        self.machine.add_transition("picked", "PICK", "PLACE")
        self.machine.add_transition("placed", "PLACE", "IDLE")

        self.machine.add_transition("fault", "*", "ERROR")



        ① 2~3대 카메라 Multi-View → Bird’s Eye Workspace Map
(“Top-Down Workspace Understanding for Manipulation + Safety”)
✔ 전체 구조 (키 포인트)
Camera1 ──────\
Camera2 ────────→ MultiCam Calibration → Unified Extrinsic (T_cam→world)
Camera3 ──────/   
                             ↓
  RGB+D → YOLO → Person/Object Detector
                             ↓
  Multi-view 3D Fusion (Triangulation, TSDF-VoxFusion)
                             ↓
  Bird’s-Eye 2D Map or 3D Occupancy (Octomap/TSDF)
                             ↓
  Robot Safety + PickPlace Planning + HRI Zone Control

✔ ROS2 패키지 구성
multicam_pkg/
 ├─ launch/
 │   └─ multicam_bird_view.launch.py
 ├─ src/
 │   ├─ multicam_sync.py          (3카메라 동기화)
 │   ├─ multicam_extrinsic_node.py (Extrinsics 자동 보정)
 │   ├─ multicam_fusion_node.py    (3D fusion + bird-eye map)
 │   └─ human_zoning_node.py       (YOLO 3D bounding box + Zone)
 └─ config/
     ├─ cam1.yaml
     ├─ cam2.yaml
     ├─ cam3.yaml
     └─ world.yaml

🟦 A. Multi-Camera Extrinsic 자동 보정

카메라 2–3대가 각각:

Cam1 → world
Cam2 → world
Cam3 → world


이 extrinsic TF를 자동으로 산출해야 함.

방법:

✔ ArUco Marker 자동 solvePnP
✔ 또는 TSDF Reconstruction + ICP Matching

(카메라끼리 보지 않아도 되는 방법)

📌 핵심 코드: multicam_extrinsic_node.py
T = cv2.solvePnP(objPoints, imgPoints)
R, _ = cv2.Rodrigues(T[1])
t = T[2]

T_cam_world = np.eye(4)
T_cam_world[:3,:3] = R
T_cam_world[:3,3] = t.ravel()

# TF broadcast
br.sendTransform(tf2_ros.TransformStamped from T_cam_world)

🟦 B. Multi-View → Bird’s-Eye Map

2~3 카메라 depth map을 TSDF로 fuse.

핵심:
TSDF Fusion(3 cam) → Voxel → Project → Top-Down Map (BEV)

📌 multicam_fusion_node.py
vol = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=0.005,
    sdf_trunc=0.03,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

for cam in [1,2,3]:
    rgbd = create_rgbd(cam)
    extr = T_world_cam[cam]
    vol.integrate(rgbd, intrinsics[cam], extr.inverse())

mesh = vol.extract_triangle_mesh()
bev_map = compute_bev(mesh)

🟦 C. Bird’s-Eye Human Zoning (YOLO + 3D Bounding Box)
입력:

YOLO bounding box (u, v)

Multi-camera → triangulated 3D person centroid

Depth = from nearest camera

3D 위치 추정:
Ray1 ∩ Ray2 ∩ Ray3 = Person 3D center

Zone 정의:
Zone A (0–1m): Emergency Stop
Zone B (1–2m): Speed Limit 25%
Zone C (2m+): Normal

📌 human_zoning_node.py
person_3d = triangulate(cam1_bbox, cam2_bbox, cam3_bbox)

distance = norm(person_3d - robot_base)

if distance < 1.0:
    pub_zone("STOP")
elif distance < 2.0:
    pub_zone("SLOW")
else:
    pub_zone("NORMAL")

🟥 ② Visual Servoing (Eye-in-Hand) + 6D GraspNet Fusion
✔ 전체 Pipeline
Wrist Camera (eye-in-hand)
      ↓
YOLO segmentation (object)
      ↓
GraspNet 6D grasp candidates
      ↓
Visual Servoing Controller (IBVS or PBVS)
      ↓
MoveIt2 Cartesian Servo (cartesian velocity command)
      ↓
Pick → Lift → Place

✔ Visual Servoing(Eye-in-Hand) 공식

Image-Based VS (IBVS)

v = -λ * L⁺ * e


Where:

e = feature error (centroid, contour, keypoints)

L⁺ = pseudo inverse of interaction matrix

v = Cartesian 6-DoF velocity command to robot

✔ 6D GraspNet과 Visual Servoing 연결 방식

GraspNet이 “이론적인” 6D grasp pose 제공

VS가 그 grasp pose로 로봇 end-effector를 보정 이동
(미세 조정 / sub-millimeter alignment)

즉:

GraspNet → 목표 자세
VS → fine alignment 수행

🟦 VS Controller ROS2 Node
📌 vs_control_node.py
import numpy as np
from geometry_msgs.msg import Twist

class VSController(Node):
    def __init__(self):
        self.sub_err = self.create_subscription(
            PixelError, "/vs/error", self.cb_err, 10)
        self.pub_cmd = self.create_publisher(
            Twist, "/servo_server/cmd_vel", 10)

        self.lambda_gain = 0.8

    def cb_err(self, msg):
        e = np.array([msg.ex, msg.ey, msg.ez, msg.erx, msg.ery, msg.erz])
        L = compute_interaction_matrix(msg)
        v = -self.lambda_gain * np.linalg.pinv(L).dot(e)

        cmd = Twist()
        cmd.linear.x  = v[0]
        cmd.linear.y  = v[1]
        cmd.linear.z  = v[2]
        cmd.angular.x = v[3]
        cmd.angular.y = v[4]
        cmd.angular.z = v[5]
        self.pub_cmd.publish(cmd)

🟦 6D GraspNet + VS “Task Policy”
Sequence:

Multi-camera BEV → pickable object 확인

Wrist-cam YOLO → object local seg

GraspNet → 6D grasp candidate

VS Align → 접촉 전 미세 정렬

Grasp → Lift → Place

🟩 전체 통합 Launch (완성판)
multicam_bird_vs_grasp.launch.py
[Camera1]
[Camera2]
[Camera3]
   → Extrinsic Calibration
   → Multi-View Fusion
   → BEV Map + Zoning (Human Safety)
[WristCam]
   → YOLO
   → GraspNet
   → Visual Servoing
[MoveIt2]
   → Grasp Execution
