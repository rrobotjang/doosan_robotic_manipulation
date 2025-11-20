import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from perception_msgs.msg import DetectedObject, Grasp6D
from cv_bridge import CvBridge

import numpy as np
import torch
from graspnetAPI import GraspNetAPI

class GraspNetNode(Node):
    def __init__(self):
        super().__init__("graspnet_node")

        self.bridge = CvBridge()
        self.api = GraspNetAPI("./graspnet_model/checkpoint")

        self.sub_obj = self.create_subscription(
            DetectedObject, "/detected_object", self.cb_obj, 10)

        self.sub_depth = self.create_subscription(
            Image, "/camera/depth/image_raw", self.cb_depth, 10)

        self.pub = self.create_publisher(Grasp6D, "/grasp_pose", 10)

        self.depth = None

    def cb_depth(self, msg):
        self.depth = self.bridge.imgmsg_to_cv2(msg, "passthrough")

    def cb_obj(self, msg):
        if self.depth is None:
            return
        
        crop_depth = self.depth[msg.y1:msg.y2, msg.x1:msg.x2]

        g = self.api.predict(crop_depth)

        grasp = Grasp6D()
        grasp.position.x = g["x"]
        grasp.position.y = g["y"]
        grasp.position.z = g["z"]
        grasp.orientation = g["quat"]  # quaternion

        self.pub.publish(grasp)


def main():
    rclpy.init()
    node = GraspNetNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
