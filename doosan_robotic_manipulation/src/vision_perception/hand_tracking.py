class HandTrackingNode(Node):
    def __init__(self):
        super().__init__("hand_tracking")
        from ultralytics import YOLO
        self.model = YOLO("yolo11n-hand-pose.pt")

        self.sub = self.create_subscription(
            Image, "/camera/color/image_raw", self.cb, 10)
        self.pub = self.create_publisher(String, "/hri/gesture", 10)

    def cb(self, msg):
        frame = ros_img_to_cv(msg)
        results = self.model(frame)[0]

        gesture = classify_gesture(results.keypoints)
        if gesture:
            self.pub.publish(String(data=gesture))
