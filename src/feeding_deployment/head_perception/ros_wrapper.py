import math
import struct
import time
from copy import deepcopy
from threading import Lock
from types import SimpleNamespace

import cv2
import message_filters
import numpy as np
import rospy
import tf2_ros
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point, TransformStamped, WrenchStamped
from scipy.spatial.transform import Rotation
from sensor_msgs import point_cloud2
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Bool, Float64, Float64MultiArray, String
from visualization_msgs.msg import Marker, MarkerArray

from feeding_deployment.head_perception.deca_perception import (
    RECORD_GOAL_POSE,
    HeadPerception,
)

VISUALIZE_DATA = False


class HeadPerceptionROSWrapper:
    def __init__(self):
        # rospy.init_node("HeadPerception")

        self.head_perception = HeadPerception()

        # Top Camera Data
        self.top_camera_lock = Lock()
        self.top_camera_header = None
        self.top_camera_color_data = None
        self.top_camera_info_data = None
        self.top_camera_depth_data = None

        # Bottom Camera Data
        self.bottom_camera_lock = Lock()
        self.bottom_camera_header = None
        self.bottom_camera_color_data = None
        self.bottom_camera_info_data = None
        self.bottom_camera_depth_data = None

        self.bridge = CvBridge()

        # Head Pose Visualisation
        self.voxel_publisher = rospy.Publisher(
            "/head_perception/voxels/marker_array", MarkerArray, queue_size=10
        )
        # self.face_publisher =  rospy.Publisher("/head_perception/face/marker_array", MarkerArray, queue_size=10)
        self.forque_publisher = rospy.Publisher(
            "/head_perception/forque/marker_array", MarkerArray, queue_size=10
        )

        self.pointcloud_publisher = rospy.Publisher(
            "/head_perception/pointcloud2", PointCloud2, queue_size=10
        )

        self.mouth_state_publisher = rospy.Publisher(
            "/head_perception/mouth_state", Bool, queue_size=10
        )
        self.head_distance_publisher = rospy.Publisher(
            "/head_distance", Float64MultiArray, queue_size=10
        )

        self.tf_buffer_lock = Lock()
        self.tfBuffer = tf2_ros.Buffer()  # Using default cache time of 10 secs
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        time.sleep(1.0)

        self.broadcaster = tf2_ros.TransformBroadcaster()

        queue_size = 1000
        self.color_image_sub = message_filters.Subscriber(
            "/camera/color/image_raw",
            Image,
            queue_size=queue_size,
            buff_size=65536 * queue_size,
        )
        self.camera_info_sub = message_filters.Subscriber(
            "/camera/color/camera_info",
            CameraInfo,
            queue_size=queue_size,
            buff_size=65536 * queue_size,
        )
        self.depth_image_sub = message_filters.Subscriber(
            "/camera/aligned_depth_to_color/image_raw",
            Image,
            queue_size=queue_size,
            buff_size=65536 * queue_size,
        )
        ts_top = message_filters.TimeSynchronizer(
            [self.color_image_sub, self.camera_info_sub, self.depth_image_sub],
            queue_size=queue_size,
        )
        ts_top.registerCallback(self.rgbdCallback)
        ts_top.enable_reset = True

        self.bottom_color_image_sub = message_filters.Subscriber(
            "/bottom_camera/color/camera_info",
            Image,
            queue_size=queue_size,
            buff_size=65536 * queue_size,
        )
        self.bottom_camera_info_sub = message_filters.Subscriber(
            "/bottom_camera/color/image_raw",
            CameraInfo,
            queue_size=queue_size,
            buff_size=65536 * queue_size,
        )
        self.bottom_depth_image_sub = message_filters.Subscriber(
            "/bottom_camera/aligned_depth_to_color/image_raw",
            Image,
            queue_size=queue_size,
            buff_size=65536 * queue_size,
        )
        ts_bottom = message_filters.TimeSynchronizer(
            [
                self.bottom_color_image_sub,
                self.bottom_camera_info_sub,
                self.bottom_depth_image_sub,
            ],
            queue_size=queue_size,
        )
        ts_bottom.registerCallback(self.bottomRgbdCallback)
        ts_bottom.enable_reset = True

    def rgbdCallback(self, rgb_image_msg, camera_info_msg, depth_image_msg):
        # print("RGB Callback")

        try:
            # Convert your ROS Image message to OpenCV2
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_image_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_image_msg, "32FC1")
        except CvBridgeError as e:
            print(e)

        with self.top_camera_lock:
            self.top_camera_color_data = rgb_image
            self.top_camera_info_data = camera_info_msg
            self.top_camera_depth_data = depth_image
            self.top_camera_header = rgb_image_msg.header

    def get_top_camera_data(self):
        with self.top_camera_lock:
            return (
                deepcopy(self.top_camera_color_data),
                deepcopy(self.top_camera_info_data),
                deepcopy(self.top_camera_depth_data),
                deepcopy(self.top_camera_header),
            )

    def bottomRgbdCallback(self, rgb_image_msg, camera_info_msg, depth_image_msg):
        # print("Bottom RGB Callback")

        try:
            # Convert your ROS Image message to OpenCV2
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_image_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_image_msg, "32FC1")
        except CvBridgeError as e:
            print(e)

        with self.bottom_camera_lock:
            self.bottom_camera_color_data = rgb_image
            self.bottom_camera_info_data = camera_info_msg
            self.bottom_camera_depth_data = depth_image
            self.bottom_camera_header = rgb_image_msg.header

    def get_bottom_camera_data(self):
        with self.bottom_camera_lock:
            return (
                deepcopy(self.bottom_camera_color_data),
                deepcopy(self.bottom_camera_info_data),
                deepcopy(self.bottom_camera_depth_data),
                deepcopy(self.bottom_camera_header),
            )

    def get_base_to_camera_transform(self, camera_info_data, use_top_camera=True):
        if use_top_camera:
            target_frame = "camera_color_optical_frame"
        else:
            target_frame = "bottom_camera_color_optical_frame"
        stamp = camera_info_data.header.stamp
        try:
            with self.tf_buffer_lock:
                # transform = self.tfBuffer.lookup_transform('base_link', 'camera_color_optical_frame', rospy.Time())
                # Rajat ToDo: The one down should be used, but it is not working for some reason
                transform = self.tfBuffer.lookup_transform(
                    "base_link",
                    target_frame,
                    rospy.Time(secs=stamp.secs, nsecs=stamp.nsecs),
                )
                return transform
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            print("Exexption finding transform between base_link and", target_frame)
            return None

    def run_head_perception(self, use_top_camera=True):

        print("Running Head Perception")
        transform = None
        while transform is None:
            if use_top_camera:
                camera_color_data, camera_info_data, camera_depth_data, _ = (
                    self.get_top_camera_data()
                )
                if camera_info_data is None:
                    continue
                transform = self.get_base_to_camera_transform(
                    camera_info_data, use_top_camera=True
                )
            else:
                camera_color_data, camera_info_data, camera_depth_data, _ = (
                    self.get_bottom_camera_data()
                )
                if camera_info_data is None:
                    continue
                transform = self.get_base_to_camera_transform(
                    camera_info_data, use_top_camera=False
                )
            # time.sleep(0.01)

        base_to_camera = np.zeros((4, 4))
        base_to_camera[:3, :3] = Rotation.from_quat(
            [
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w,
            ]
        ).as_matrix()
        base_to_camera[:3, 3] = np.array(
            [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ]
        ).reshape(1, 3)
        base_to_camera[3, 3] = 1

        run_deca_start_time = time.time()
        print("Calling DECA")
        (
            landmarks2d,
            landmarks3d,
            viz_image,
            mouth_state,
            average_head_point,
            forque_target_pose,
            visualization_points_world_frame,
            reference_neck_frame,
            neck_frame,
        ) = self.head_perception.run_deca(
            camera_color_data,
            camera_info_data,
            camera_depth_data,
            base_to_camera,
            debug_print=False,
            visualize=VISUALIZE_DATA,
        )
        run_deca_end_time = time.time()
        print("Run DECA time: ", run_deca_end_time - run_deca_start_time)

        if landmarks2d is not None:
            self.mouth_state_publisher.publish(mouth_state)

            head_distance_msg = Float64MultiArray()
            head_distance_msg.data = [
                average_head_point[0],
                average_head_point[1],
                average_head_point[2],
            ]
            self.head_distance_publisher.publish(head_distance_msg)

            self.visualizeForque(forque_target_pose)
            self.visualizeVoxels(visualization_points_world_frame)

            self.updateTF("base_link", "forque_end_effector_target", forque_target_pose)
            self.updateTF("base_link", "head_pose", neck_frame)
            self.updateTF("base_link", "reference_head_pose", reference_neck_frame)

        return forque_target_pose

    def updateTF(self, source_frame, target_frame, pose):

        # return #do nothing to surpress warnings with rosbags

        t = TransformStamped()

        t.header.stamp = rospy.Time.now()
        t.header.frame_id = source_frame
        t.child_frame_id = target_frame

        t.transform.translation.x = pose[0][3]
        t.transform.translation.y = pose[1][3]
        t.transform.translation.z = pose[2][3]

        R = Rotation.from_matrix(pose[:3, :3]).as_quat()
        t.transform.rotation.x = R[0]
        t.transform.rotation.y = R[1]
        t.transform.rotation.z = R[2]
        t.transform.rotation.w = R[3]

        self.broadcaster.sendTransform(t)

    def visualizeForque(self, pose):

        # pose = self.base_to_camera @ pose

        markerArray = MarkerArray()

        forque_marker = Marker()
        forque_marker.header.seq = 0
        forque_marker.header.stamp = rospy.Time.now()
        if RECORD_GOAL_POSE:
            forque_marker.header.frame_id = "camera_color_optical_frame"
        else:
            forque_marker.header.frame_id = "base_link"
        forque_marker.ns = "forque_marker"
        forque_marker.id = 1
        forque_marker.type = forque_marker.MESH_RESOURCE  # CUBE LIST
        forque_marker.mesh_resource = "file:///home/isacc/deployment_ws/src/ada_description/kortex_description/tools/forque/forque_tip.stl"
        forque_marker.mesh_use_embedded_materials = True
        forque_marker.action = forque_marker.ADD  # ADD
        forque_marker.lifetime = rospy.Duration()

        forque_marker.color.a = 1.0
        forque_marker.color.r = 0.5
        forque_marker.color.g = 0.5
        forque_marker.color.b = 0.5

        forque_marker.pose.position.x = pose[0][3]
        forque_marker.pose.position.y = pose[1][3]
        forque_marker.pose.position.z = pose[2][3]

        R = Rotation.from_matrix(pose[:3, :3]).as_quat()
        forque_marker.pose.orientation.x = R[0]
        forque_marker.pose.orientation.y = R[1]
        forque_marker.pose.orientation.z = R[2]
        forque_marker.pose.orientation.w = R[3]

        forque_marker.scale.x = 0.001
        forque_marker.scale.y = 0.001
        forque_marker.scale.z = 0.001

        markerArray.markers.append(forque_marker)

        self.forque_publisher.publish(markerArray)

    def visualizeVoxels(self, voxels, namespace="visualize_voxels"):

        # print(voxels)

        markerArray = MarkerArray()

        marker = Marker()
        marker.header.seq = 0
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = "base_link"
        marker.ns = namespace
        marker.id = 1
        marker.type = 6
        # CUBE LIST
        marker.action = 0
        # ADD
        marker.lifetime = rospy.Duration()
        marker.scale.x = 0.01
        marker.scale.y = 0.01
        marker.scale.z = 0.01
        marker.color.r = 1
        marker.color.g = 1
        marker.color.b = 0
        marker.color.a = 1

        for i in range(voxels.shape[0]):

            point = Point()
            point.x = voxels[i, 0]
            point.y = voxels[i, 1]
            point.z = voxels[i, 2]

            marker.points.append(point)

        markerArray.markers.append(marker)

        self.voxel_publisher.publish(markerArray)


if __name__ == "__main__":
    rospy.init_node("head_perception", anonymous=True)
    head_perception_ros_wrapper = HeadPerceptionROSWrapper()
    time.sleep(2.0)  # let the buffers fill up
    while not rospy.is_shutdown():
        head_perception_ros_wrapper.run_head_perception()
        # rospy.sleep(0.1)
