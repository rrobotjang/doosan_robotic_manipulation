#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np

class ObjectTrackingNode(Node):
    def __init__(self):
        super().__init__('object_tracking_node')

        self.bridge = CvBridge()
        self.tracker = None
        self.tracking = False  # tracker 활성화 여부
        self.last_bbox = None

        # Topics
        self.sub_img = self.create_subscription(Image,
                        "/camera/color/image_raw", self.img_cb, 10)

        # YOLO에서 bbox 들어옴
        self.sub_yolo = self.create_subscription(String,
                        "/yolo/bbox", self.yolo_cb, 10)

        self.pub_pose = self.create_publisher(PoseStamped,
                        "/object_pose", 10)

    def yolo_cb(self, msg):
        """ YOLO가 새 bbox를 보내주면 tracker 초기화 """
        bbox = eval(msg.data)   # ex: "(x, y, w, h)"
        self.last_bbox = bbox
        self.tracking = False   # 다음 프레임에서 tracker 재생성

    def img_cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg)

        # YOLO bbox가 들어오면 tracker 초기화
        if self.last_bbox is not None and not self.tracking:
            self.tracker = cv2.TrackerCSRT_create()
            self.tracker.init(frame, tuple(self.last_bbox))
            self.tracking = True
            return

        # tracker 동작 중 → update 수행
        if self.tracking:
            success, bbox = self.tracker.update(frame)
            if success:
                x, y, w, h = [int(i) for i in bbox]
                cx = x + w/2
                cy = y + h/2

                # depth 기반 z 계산 (예시)
                z = self.get_depth(cx, cy)

                pose = PoseStamped()
                pose.header.frame_id = "camera_link"
                pose.pose.position.x = cx
                pose.pose.position.y = cy
                pose.pose.position.z = float(z)

                self.pub_pose.publish(pose)
            else:
                # tracking 실패 → YOLO 신호 대기
                self.tracking = False

    def get_depth(self, x, y):
        # 이 부분은 사용 센서에 맞춰 구현
        return 0.20

def main():
    rclpy.init()
    node = ObjectTrackingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
