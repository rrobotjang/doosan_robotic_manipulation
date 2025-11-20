#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from sensor_msgs.msg import Image

class PickPlaceNode(Node):
    def __init__(self):
        super().__init__("pick_place_node")

        # Subscribers
        self.obj_sub = self.create_subscription(
            PoseStamped, "/object_pose", self.obj_cb, 10)
        self.gesture_sub = self.create_subscription(
            String, "/hri/gesture", self.gesture_cb, 10)

        # Publishers
        self.cmd_pub = self.create_publisher(String, "/doosan/cmd", 10)

        self.current_target = None
        self.hri_state = "IDLE"

    def obj_cb(self, msg):
        self.get_logger().info(f"Object detected: {msg.pose}")
        self.current_target = msg.pose
        if self.hri_state == "READY":
            self.do_pick_and_place()

    def gesture_cb(self, msg):
        gesture = msg.data
        self.get_logger().info(f"HRI Gesture: {gesture}")

        if gesture == "START_PICK":
            self.hri_state = "READY"
        elif gesture == "STOP":
            self.hri_state = "STOPPED"

    def do_pick_and_place(self):
        if not self.current_target: 
            return
        
        self.get_logger().info("Executing pick & place...")
        cmd = f"PICK_PLACE:{self.current_target.position.x}," \
              f"{self.current_target.position.y}," \
              f"{self.current_target.position.z}"
        self.cmd_pub.publish(String(data=cmd))


def main():
    rclpy.init()
    node = PickPlaceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
