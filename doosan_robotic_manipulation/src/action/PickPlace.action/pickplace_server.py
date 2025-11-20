#!/usr/bin/env python3
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node

from moveit_msgs.msg import MoveItErrorCodes
from geometry_msgs.msg import Pose
from my_interfaces.action import PickPlace

import moveit_commander
import time


class PickPlaceServer(Node):
    def __init__(self):
        super().__init__("pickplace_server")

        moveit_commander.roscpp_initialize([])

        self.arm = moveit_commander.MoveGroupCommander("arm")
        self.gripper = moveit_commander.MoveGroupCommander("gripper")

        self._action_server = ActionServer(
            self,
            PickPlace,
            "pick_place",
            self.execute_cb)

        self.get_logger().info("Pick&Place Action Server Ready")

    def execute_cb(self, goal):

        target = goal.request.target

        # Approach
        approach = Pose()
        approach.position.x = target.position.x
        approach.position.y = target.position.y
        approach.position.z = target.position.z + 0.10

        self.arm.set_pose_target(approach)
        res = self.arm.go(wait=True)
        if not res:
            return PickPlace.Result(success=False, message="Fail approach")

        # Down
        down = target
        self.arm.set_pose_target(down)
        self.arm.go(wait=True)

        # Grip
        self.gripper.set_named_target("close")
        self.gripper.go(wait=True)

        # Lift
        lift = Pose()
        lift.position.x = target.position.x
        lift.position.y = target.position.y
        lift.position.z = target.position.z + 0.20
        self.arm.set_pose_target(lift)
        self.arm.go(wait=True)

        # Place
        place = Pose()
        place.position.x = 0.45
        place.position.y = 0.0
        place.position.z = 0.25
        self.arm.set_pose_target(place)
        self.arm.go(wait=True)

        # Release
        self.gripper.set_named_target("open")
        self.gripper.go(wait=True)

        return PickPlace.Result(success=True, message="Done")


def main():
    rclpy.init()
    node = PickPlaceServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
