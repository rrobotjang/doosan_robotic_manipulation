#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
import cv2
from ultralytics import YOLO
from cv_bridge import CvBridge

class HandTrackingNode(Node):
    def __init__(self):
        super().__init__("hand_tracking_node")

        self.model = YOLO("yolo11n-hand-pose.pt")
        self.bridge = CvBridge()

        self.sub = self.create_subscription(
            Image, "/camera/color/image_raw", self.cb, 10)
        self.pub = self.create_publisher(String, "/hri/gesture", 10)

    def cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg)

        results = self.model(frame, verbose=False)[0]

        if not results.keypoints:
            return

        kps = results.keypoints[0].xy.cpu().numpy()

        gesture = self.classify_gesture(kps)

        if gesture:
            self.pub.publish(String(data=gesture))
            self.get_logger().info(f"Gesture: {gesture}")

    def classify_gesture(self, kps):
        # 손가락 끝 인덱스
        TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}

        # F(접힘) = TIP.y > knuckle.y
        def is_folded(tip, knuckle):
            return kps[tip][1] > kps[knuckle][1]

        thumb_fold = is_folded(TIP["thumb"], 3)
        index_fold = is_folded(TIP["index"], 5)
        middle_fold = is_folded(TIP["middle"], 9)
        ring_fold = is_folded(TIP["ring"], 13)
        pinky_fold = is_folded(TIP["pinky"], 17)

        # ✊ 주먹
        if thumb_fold and index_fold and middle_fold and ring_fold and pinky_fold:
            return "STOP"

        # 🖐 손바닥
        if not index_fold and not middle_fold and not ring_fold and not pinky_fold:
            return "OPEN"

        # 👉 집게손
        if (not index_fold) and middle_fold and ring_fold:
            return "START_PICK"

        # 👍 엄지척
        if (not thumb_fold) and index_fold and middle_fold and ring_fold:
            return "OK"

        return None


def main():
    rclpy.init()
    node = HandTrackingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
