#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from cv_bridge import CvBridge

import cv2
import numpy as np

from ultralytics import YOLO

# ByteTrack modules
from bytetrack.byte_tracker import BYTETracker
from bytetrack.utils import STrack

class YoloByteTrackNode(Node):
    def __init__(self):
        super().__init__('yolo_bytetrack_node')

        self.bridge = CvBridge()
        self.model = YOLO("yolov8s.pt")

        self.tracker = BYTETracker()

        self.sub_cam = self.create_subscription(
            Image, "/camera/color/image_raw", self.cb, 10)

        self.pub_pose = self.create_publisher(
            PoseStamped, "/object_pose", 10)

    def cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg)

        # YOLO inference
        yolo_out = self.model(frame, conf=0.6, verbose=False)[0]

        dets = []
        for b in yolo_out.boxes:
            x1, y1, x2, y2 = b.xyxy[0]
            conf = float(b.conf[0])
            cls = int(b.cls[0])

            # only target class filtering (예: bottle)
            if cls in [39, 0, 41]:
                dets.append([x1, y1, x2, y2, conf, cls])

        online_targets = self.tracker.update(np.array(dets))

        for track in online_targets:
            # 트래커가 유지하는 좌표
            x1, y1, x2, y2 = track.tlwh_to_xyxy()

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            z = self.get_depth(cx, cy)  # depth → z 계산

            pose = PoseStamped()
            pose.header.frame_id = "camera_link"

            pose.pose.position.x = self.pixel_to_cam_x(cx, z)
            pose.pose.position.y = self.pixel_to_cam_y(cy, z)
            pose.pose.position.z = z

            self.pub_pose.publish(pose)

    # depth image 사용하는 경우 이 함수 구현
    def get_depth(self, x, y):
        # 실제 구현은 depth image subscribe해서 use!
        return 0.28

    # Camera Intrinsics
    fx = 615; fy = 615; cx0 = 320; cy0 = 240

    def pixel_to_cam_x(self, u, depth):
        return (u - self.cx0) * depth / self.fx

    def pixel_to_cam_y(self, v, depth):
        return (v - self.cy0) * depth / self.fy


def main():
    rclpy.init()
    node = YoloByteTrackNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

