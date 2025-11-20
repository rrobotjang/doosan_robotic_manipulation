#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Twist
from visual_servoing_pkg.vs_controller import VisualServoController

class VSControlNode(Node):
    def __init__(self):
        super().__init__("vs_control")

        self.controller = VisualServoController(fx=615, fy=615, cx=320, cy=240)

        self.sub = self.create_subscription(
            PoseStamped, "/object_pose", self.cb, 10)

        self.pub_cmd = self.create_publisher(
            Twist, "/dsr01/arm/cartesian_vel", 10)

    def cb(self, msg):
        u = msg.pose.position.x_pixel
        v = msg.pose.position.y_pixel
        z = msg.pose.position.z

        twist = Twist()
        v_cart = self.controller.compute_control(u, v, z)
        twist.linear.x = float(v_cart[0])
        twist.linear.y = float(v_cart[1])
        twist.linear.z = float(v_cart[2])

        twist.angular.x = float(v_cart[3])
        twist.angular.y = float(v_cart[4])
        twist.angular.z = float(v_cart[5])

        self.pub_cmd.publish(twist)


def main():
    rclpy.init()
    node = VSControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
