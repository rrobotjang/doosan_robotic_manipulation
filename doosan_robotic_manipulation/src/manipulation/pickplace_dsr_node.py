#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray, String


class PickPlaceDSR(Node):
    def __init__(self):
        super().__init__("pickplace_dsr_node")

        # Subscribe Object Pose
        self.sub_pose = self.create_subscription(
            PoseStamped, "/object_pose", self.cb, 10)

        # Publish Doosan command
        self.pub_move = self.create_publisher(
            Float64MultiArray, "/dsr01/arm/move_line", 10)

        self.pub_grip = self.create_publisher(
            String, "/dsr01/gripper/control", 10)

    def cb(self, msg):
        pose = msg.pose
        self.get_logger().info(f"Target: {pose.position}")

        self.pick_and_place(pose)

    def pick_and_place(self, p):

        # 1) Approach
        self.move_line(p.position.x, p.position.y, p.position.z + 0.10,
                       180, 0, 180)

        # 2) Down
        self.move_line(p.position.x, p.position.y, p.position.z,
                       180, 0, 180)

        # 3) Grip
        self.pub_grip.publish(String(data="close"))
        self.get_logger().info("close gripper")

        # 4) Lift
        self.move_line(p.position.x, p.position.y, p.position.z + 0.20,
                       180, 0, 180)

        # 5) Drop zone
        self.move_line(0.45, 0.0, 0.30, 180, 0, 180)

        self.pub_grip.publish(String(data="open"))
        self.get_logger().info("open gripper")

    def move_line(self, x, y, z, rx, ry, rz):
        msg = Float64MultiArray()
        msg.data = [x, y, z, rx, ry, rz]
        self.pub_move.publish(msg)


def main():
    rclpy.init()
    node = PickPlaceDSR()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
