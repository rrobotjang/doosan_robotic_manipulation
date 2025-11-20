import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import OccupancyGrid

import open3d as o3d
import numpy as np
from cv_bridge import CvBridge

class TSDFBuilder(Node):
    def __init__(self):
        super().__init__("tsdf_builder")

        self.bridge = CvBridge()

        self.sub_depth = self.create_subscription(
            Image, "/camera/depth/image_raw", self.cb_depth, 10)

        self.integrator = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=0.004,
            sdf_trunc=0.02,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

        self.depth_intrinsics = None

    def cb_depth(self, msg):
        depth = self.bridge.imgmsg_to_cv2(msg, "passthrough")

        if self.depth_intrinsics is None:
            return

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.zeros_like(depth)),
            o3d.geometry.Image(depth),
            depth_scale=1000.0,
            depth_trunc=1.2,
            convert_rgb_to_intensity=False
        )

        extrinsic = np.eye(4)
        self.integrator.integrate(rgbd, self.depth_intrinsics, extrinsic)

    def save_mesh(self):
        mesh = self.integrator.extract_triangle_mesh()
        mesh.filter_smooth_simple(number_of_iterations=5)
        o3d.io.write_triangle_mesh("/ros2_ws/collision_env.obj", mesh)
        print("Saved collision_env.obj")


def main():
    rclpy.init()
    node = TSDFBuilder()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_mesh()

    node.destroy_node()
    rclpy.shutdown()
