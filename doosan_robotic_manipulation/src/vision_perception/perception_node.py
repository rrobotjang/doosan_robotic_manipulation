#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import numpy as np
import cv2

from ultralytics import YOLO
from filterpy.kalman import KalmanFilter

from bytetrack.byte_tracker import BYTETracker


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")

        self.bridge = CvBridge()

        # YOLO
        self.model = YOLO("yolov8s.pt")

        # ByteTrack
        self.tracker = BYTETracker()

        # Kalman Filters per ID
        self.kalman_filters = {}

        # Camera intrinsics
        self.fx = 615; self.fy = 615; self.cx = 320; self.cy = 240

        # Subs
        self.sub_rgb = self.create_subscription(
            Image, "/camera/color/image_raw", self.rgb_cb, 10)
        self.sub_depth = self.create_subscription(
            Image, "/camera/depth/image_raw", self.depth_cb, 10)

        # Pub
        self.pub_pose = self.create_publisher(PoseStamped, "/object_pose", 10)

        self.depth_frame = None

    def depth_cb(self, msg):
        self.depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def rgb_cb(self, msg):
        if self.depth_frame is None:
            return
        frame = self.bridge.imgmsg_to_cv2(msg)

        yolo_out = self.model(frame, conf=0.6, verbose=False)[0]

        dets = []
        for b in yolo_out.boxes:
            cls = int(b.cls[0])
            if cls not in [39, 41, 0]:  # cup, bottle 등
                continue
            x1, y1, x2, y2 = b.xyxy[0]
            conf = float(b.conf[0])
            dets.append([x1, y1, x2, y2, conf, cls])

        tracks = self.tracker.update(np.array(dets))

        for t in tracks:
            x1, y1, x2, y2 = t.tlwh_to_xyxy()
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            z = float(self.depth_frame[cy, cx]) / 1000.0
            if z <= 0:
                continue

            X = (cx - self.cx) * z / self.fx
            Y = (cy - self.cy) * z / self.fy

            obj_id = t.track_id

            # Kalman Filter per ID
            kf = self.ensure_kalman(obj_id)
            smoothed = kf.predict()
            kf.update(np.array([X, Y, z]))

            pose = PoseStamped()
            pose.header.frame_id = "camera_link"
            pose.pose.position.x = kf.x[0]
            pose.pose.position.y = kf.x[1]
            pose.pose.position.z = kf.x[2]

            self.pub_pose.publish(pose)

    def ensure_kalman(self, obj_id):
        if obj_id in self.kalman_filters:
            return self.kalman_filters[obj_id]

        kf = KalmanFilter(dim_x=6, dim_z=3)
        kf.F = np.array([[1,0,0,1,0,0],
                         [0,1,0,0,1,0],
                         [0,0,1,0,0,1],
                         [0,0,0,1,0,0],
                         [0,0,0,0,1,0],
                         [0,0,0,0,0,1]])
        kf.H = np.array([[1,0,0,0,0,0],
                         [0,1,0,0,0,0],
                         [0,0,1,0,0,0]])
        kf.P *= 0.1
        kf.R *= 0.01
        kf.Q *= 0.001

        self.kalman_filters[obj_id] = kf
        return kf


def main():
    rclpy.init()
    node = PerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
