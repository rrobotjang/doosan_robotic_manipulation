#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import whisper
import torch
from transformers import pipeline

class SpeechCommandNode(Node):
    def __init__(self):
        super().__init__("speech_cmd")

        # Whisper ASR
        self.model = whisper.load_model("small")

        # Intent classifier (fine-tuned for robot commands)
        self.intent = pipeline("text-classification", model="bert-hri-intent")

        self.pub_cmd = self.create_publisher(String, "/hri/cmd", 10)

        self.get_logger().info("HRI Speech Command Ready")

        # Microphone streaming timer
        self.create_timer(0.5, self.listen)

    def listen(self):
        try:
            audio = whisper.load_audio("mic_input.wav")
            audio = whisper.pad_or_trim(audio)

            result = self.model.transcribe(audio, language='ko')
            text = result["text"].strip()

            if len(text) == 0:
                return

            intent = self.intent(text)[0]["label"]

            # Intent → command mapping
            cmd = String()
            if intent == "START_PICK":
                cmd.data = "START"
            elif intent == "STOP":
                cmd.data = "STOP"
            elif intent == "RESUME":
                cmd.data = "RESUME"
            elif intent == "CANCEL":
                cmd.data = "CANCEL"
            elif intent == "SLOW":
                cmd.data = "SLOW"
            elif intent == "SPEEDUP":
                cmd.data = "FAST"

            self.pub_cmd.publish(cmd)
            self.get_logger().info(f"ASR: {text} → Intent: {intent}")

        except Exception as e:
            pass


def main():
    rclpy.init()
    node = SpeechCommandNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
