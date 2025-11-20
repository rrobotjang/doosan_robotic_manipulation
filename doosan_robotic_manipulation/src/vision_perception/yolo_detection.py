class YoloDetector(Node):
    def __init__(self):
        super().__init__("yolo_detector")
        from ultralytics import YOLO
        self.model = YOLO("yolov8s.pt")

        self.image_sub = self.create_subscription(
            Image, "/camera/color/image_raw", self.img_cb, 10)

        self.obj_pub = self.create_publisher(PoseStamped, "/object_pose", 10)

    def img_cb(self, msg):
        frame = ros_img_to_cv(msg)
        results = self.model(frame)[0]

        for box in results.boxes:
            cls_id = int(box.cls)
            if cls_id in [0, 39]:  # bottle, cup 등
                x, y = box.xyxy[0][:2]
                depth = get_depth(x, y)
                pose = depth_to_pose(x, y, depth)
                self.obj_pub.publish(pose)
