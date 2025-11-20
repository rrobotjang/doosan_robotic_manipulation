#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import requests
import json

class VLMQueryNode(Node):
    def __init__(self):
        super().__init__("vlm_query_node")
        self.sub = self.create_subscription(
            String, "/user_command", self.cb, 10)

        self.pub = self.create_publisher(String, "/filtered_target_class", 10)

    def cb(self, msg):
        text = msg.data
        vlm_result = self.query_vlm(text)

        target = self.extract_target(vlm_result)
        self.pub.publish(String(data=target))
        self.get_logger().info(f"VLM target: {target}")

    def query_vlm(self, text):
        req = {"text": text}
        r = requests.post(
            "http://vlm-server:8080/v1/interpret",
            data=json.dumps(req),
            headers={"Content-Type": "application/json"}
        )
        return r.json()

    def extract_target(self, vlm_json):
        # 예시: {"target": "blue object", "action": "pick and place"}
        return vlm_json.get("target", "object")


def main():
    rclpy.init()
    node = VLMQueryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
