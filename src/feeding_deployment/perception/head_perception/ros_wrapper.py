import math
import struct
import time
from copy import deepcopy
from threading import Lock
from types import SimpleNamespace

import cv2
import argparse
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

from feeding_deployment.perception.head_perception.deca_perception import (
    HeadPerception,
)

class HeadPerceptionROSWrapper:
    def __init__(self, record_goal_pose=False):
        # rospy.init_node("HeadPerception")

        self.head_perception = HeadPerception(record_goal_pose)

        # Top Camera Data
        self.camera_lock = Lock()
        self.camera_header = None
        self.camera_color_data = None
        self.camera_info_data = None
        self.camera_depth_data = None

        self.bridge = CvBridge()

        # Head Pose Visualisation
        self.voxel_publisher = rospy.Publisher(
            "/head_perception/voxels/marker_array", MarkerArray, queue_size=10
        )

        self.tool_publisher = rospy.Publisher(
            "/head_perception/tool/marker_array", MarkerArray, queue_size=10
        )

        self.noisy_reading_publisher = rospy.Publisher(
            "/head_perception/unexpected", Bool, queue_size=10
        )

        self.tf_buffer_lock = Lock()
        self.tfBuffer = tf2_ros.Buffer()  # Using default cache time of 10 secs
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

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

        self.filter_noisy_readings = False
        self.filter_noisy_readings_sub = rospy.Subscriber(
            "/head_perception/set_filter_noisy_readings", Bool, self.setFilterNoisyReadingsCallback, queue_size=1
        )
    
        time.sleep(2.0) # sleep until all subscribers are registered

    def setFilterNoisyReadingsCallback(self, msg):
        self.filter_noisy_readings = msg.data

    def rgbdCallback(self, rgb_image_msg, camera_info_msg, depth_image_msg):
        # print("RGB Callback")

        try:
            # Convert your ROS Image message to OpenCV2
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_image_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_image_msg, "32FC1")
        except CvBridgeError as e:
            print(e)

        with self.camera_lock:
            self.camera_color_data = rgb_image
            self.camera_info_data = camera_info_msg
            self.camera_depth_data = depth_image
            self.camera_header = rgb_image_msg.header

    def get_camera_data(self):
        with self.camera_lock:
            return (
                deepcopy(self.camera_color_data),
                deepcopy(self.camera_info_data),
                deepcopy(self.camera_depth_data),
                deepcopy(self.camera_header),
            )

    def get_base_to_camera_transform(self, camera_info_data):
        target_frame = "camera_color_optical_frame"
        stamp = camera_info_data.header.stamp
        try:
            with self.tf_buffer_lock:
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
            # print("Exception finding transform between base_link and", target_frame)
            return None

    def run_head_perception(self, visualize=False):

        # print("Running Head Perception")
        transform = None
        while transform is None:
            camera_color_data, camera_info_data, camera_depth_data, _ = (
                self.get_camera_data()
            )
            if camera_info_data is None:
                # print("No camera data")
                time.sleep(0.01)
                continue
            transform = self.get_base_to_camera_transform(camera_info_data)

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
        head_perception_data = self.head_perception.run_deca(
            camera_color_data,
            camera_info_data,
            camera_depth_data,
            base_to_camera,
            debug_print=False,
            visualize=visualize,
            filter_noisy_readings=self.filter_noisy_readings,
        )
        run_deca_end_time = time.time()
        # print("Run DECA time: ", run_deca_end_time - run_deca_start_time)

        if head_perception_data is not None:

            if self.filter_noisy_readings: # do not shutdown robot if warm starting / kill_on_noisy_reading is False
                if head_perception_data["noisy_reading"] is not None:
                    self.noisy_reading_publisher.publish(head_perception_data["noisy_reading"])

            self.visualizeToolTipTarget(head_perception_data["tool_tip_target_pose"])
            self.visualizeVoxels(head_perception_data["visualization_points_world_frame"])

            if self.head_perception.record_goal_pose:
                self.updateTF("camera_color_optical_frame", "tool_tip_target", head_perception_data["tool_tip_target_pose"])
            else:
                self.updateTF("base_link", "tool_tip_target", head_perception_data["tool_tip_target_pose"])
            self.updateTF("base_link", "head_pose", head_perception_data["neck_frame"])
            self.updateTF("base_link", "reference_head_pose", head_perception_data["reference_neck_frame"])

            return {
                "head_pose": head_perception_data["head_pose"],
                "face_keypoints": head_perception_data["landmarks2d"],
                "tool_tip_target_pose": head_perception_data["tool_tip_target_pose"],
            }

        else:
            if self.filter_noisy_readings:
                self.noisy_reading_publisher.publish(Bool(data=True))
            
            return None

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

    def visualizeToolTipTarget(self, pose):

        # pose = self.base_to_camera @ pose

        markerArray = MarkerArray()

        tool_marker = Marker()
        tool_marker.header.seq = 0
        tool_marker.header.stamp = rospy.Time.now()
        if self.head_perception.record_goal_pose:
            tool_marker.header.frame_id = "camera_color_optical_frame"
        else:
            tool_marker.header.frame_id = "base_link"
        # print(" --- Header frame id: ", tool_marker.header.frame_id)
        tool_marker.ns = "tool_marker"
        tool_marker.id = 1
        tool_marker.type = tool_marker.MESH_RESOURCE  # CUBE LIST
        if self.head_perception.tool == "fork":
            tool_marker.mesh_resource = "file:////home/isacc/deployment_ws/src/kortex_description/tools/feeding_tool/tool_tip.stl"
        elif self.head_perception.tool == "wipe":
            tool_marker.mesh_resource = "file:////home/isacc/deployment_ws/src/kortex_description/tools/wiping_tool/tool_tip.stl"
        elif self.head_perception.tool == "drink":
            tool_marker.mesh_resource = "file:////home/isacc/deployment_ws/src/kortex_description/tools/drinking_tool/tool_tip.stl"
        else:
            raise ValueError(f"Mesh does not exist for tool: {self.head_perception}")
        
        tool_marker.mesh_use_embedded_materials = True
        tool_marker.action = tool_marker.ADD  # ADD
        tool_marker.lifetime = rospy.Duration()

        tool_marker.color.a = 1.0
        tool_marker.color.r = 0.5
        tool_marker.color.g = 0.5
        tool_marker.color.b = 0.5

        tool_marker.pose.position.x = pose[0][3]
        tool_marker.pose.position.y = pose[1][3]
        tool_marker.pose.position.z = pose[2][3]

        R = Rotation.from_matrix(pose[:3, :3]).as_quat()
        tool_marker.pose.orientation.x = R[0]
        tool_marker.pose.orientation.y = R[1]
        tool_marker.pose.orientation.z = R[2]
        tool_marker.pose.orientation.w = R[3]

        tool_marker.scale.x = 1.0
        tool_marker.scale.y = 1.0
        tool_marker.scale.z = 1.0

        markerArray.markers.append(tool_marker)

        self.tool_publisher.publish(markerArray)

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

    def save_tool_tip_transform(self, tool):

        rate = rospy.Rate(1000.0)
        while not rospy.is_shutdown():
            try:
                print("Looking for transform")
                with self.tf_buffer_lock:
                    transform = self.tfBuffer.lookup_transform('camera_color_optical_frame', tool + '_tip', rospy.Time())
                    break
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                try:
                    rate.sleep()
                except Exception as inst:
                    print(inst)
                continue

        tool_planar_tip_pose = np.zeros((4,4))
        tool_planar_tip_pose[:3,:3] = Rotation.from_quat([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]).as_matrix()
        tool_planar_tip_pose[:3,3] = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]).reshape(1,3)
        tool_planar_tip_pose[3,3] = 1

        self.head_perception.save_tool_tip_transform(tool, tool_planar_tip_pose)

    def set_tool(self, tool):
        self.head_perception.set_tool(tool)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--record_goal_pose", action="store_true")
    parser.add_argument("--tool", type=str, default="fork")
    parser.add_argument("--set_tool_tip_transform", action="store_true")
    parser.add_argument("--visualize", action="store_true")

    args = parser.parse_args()

    rospy.init_node("head_perception", anonymous=True)

    head_perception = HeadPerception(record_goal_pose=args.record_goal_pose)
    head_perception_ros_wrapper = HeadPerceptionROSWrapper(head_perception)
    time.sleep(2.0)  # let the buffers fill up

    if args.set_tool_tip_transform:
        head_perception_ros_wrapper.save_tool_tip_transform(args.tool)
    head_perception_ros_wrapper.set_tool(args.tool)
    while not rospy.is_shutdown():
        head_perception_ros_wrapper.run_head_perception(visualize=args.visualize)
        # rospy.sleep(0.1)
