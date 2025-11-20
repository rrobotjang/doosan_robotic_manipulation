#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped
from my_interfaces.action import PickPlace


class PickPlaceClient(Node):
    def __init__(self):
        super().__init__("pickplace_client")

        self.client = ActionClient(self, PickPlace, "pick_place")
        self.sub_pose = self.create_subscription(
            PoseStamped, "/object_pose", self.pose_cb, 10)

        self.hri_cmd = "START"

    def pose_cb(self, msg):
        if self.hri_cmd != "START":
            return

        self.send_goal(msg.pose)

    def send_goal(self, pose):
        goal_msg = PickPlace.Goal()
        goal_msg.target = pose

        self.client.wait_for_server()
        self.client.send_goal_async(goal_msg)


def main():
    rclpy.init()
    node = PickPlaceClient()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
