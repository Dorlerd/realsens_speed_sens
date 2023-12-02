# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
import math
from rclpy.node import Node
import filterpy.kalman
import filterpy.common
import numpy as np

from cv_bridge import CvBridge
from std_msgs.msg import String,Float32
from sensor_msgs.msg import Image,TimeReference
from geometry_msgs.msg import Vector3,Quaternion
import open3d as o3d
from rclpy.qos import qos_profile_sensor_data, QoSProfile
import numpy as np


class Subscriber(Node):

    def __init__(self):
        qos_profile = QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                                    history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                                    depth=5)
        super().__init__('odom_cam')
        self.br = CvBridge()
        self.img_rgb = 0
        self.img_depth = 0
        self.img_rgb_prev = 0
        self.img_depth_prev = 0
        self.tf = 0
        self.first = True
        self.velocity_cam = self.create_publisher(Quaternion, 'velocity', 10)
        self.steering = self.create_publisher(Vector3, 'steering', 10)
        self.pose = self.create_publisher(Vector3, 'pose', 10)
        self.subscription = self.create_subscription(Image,'/depth_camera/depth/image_raw',self.save_img,qos_profile=qos_profile_sensor_data)
        self.subscription2 = self.create_subscription(Image,'/depth_camera/image_raw',self.odom_calc,qos_profile=qos_profile_sensor_data)
    def save_img(self,msg):
        self.img_depth= self.br.imgmsg_to_cv2(msg,'passthrough')
        self.img_depth = np.array(self.img_depth)
        print(self.img_depth.shape)
        self.img_depth = o3d.geometry.Image(self.img_depth.astype(np.float32))
    def odom_calc(self,msg):
        self.img_rgb = self.br.imgmsg_to_cv2(msg,"rgb8")
        self.img_rgb = np.array(self.img_rgb )
        print(self.img_rgb.shape)
        self.img_rgb = o3d.geometry.Image(self.img_rgb.astype(np.uint8))
        if self.first:
            self.img_rgb_prev = self.img_rgb
            self.img_depth_prev = self.img_depth
            self.first = False 
            return
        
        source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(self.img_rgb_prev, self.img_depth_prev)
        target_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(self.img_rgb, self.img_depth)
        option = o3d.pipelines.odometry.OdometryOption()
        odo_init = np.identity(4)
        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
        pinhole_camera_intrinsic.intrinsic_matrix = [[554.254691191187,    0, 320.5],
                                    [  0, 554.254691191187,  240.5],
                                    [  0,   0,   1 ]]
        self.img_rgb_prev=self.img_rgb
        self.img_depth_prev=self.img_depth
        [success_hybrid_term, trans_hybrid_term,
 info] = o3d.pipelines.odometry.compute_rgbd_odometry(
     source_rgbd_image, target_rgbd_image, pinhole_camera_intrinsic, odo_init,
     o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
        print(trans_hybrid_term)
        
        

def main(args=None):
    rclpy.init(args=args)

    subscriber = Subscriber()

    rclpy.spin(subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()





