import os
import sys
import cv2
import numpy as np
import time
import math
from scipy.spatial.transform import Rotation

import rospy
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import tf2_ros
from geometry_msgs.msg import Point, Pose
from visualization_msgs.msg import MarkerArray, Marker

from feeding_deployment.control.robot_controller.arm_client import ArmInterfaceClient
from feeding_deployment.control.robot_controller.command_interface import (
    CartesianCommand,
    JointCommand,
    CloseGripperCommand,
    OpenGripperCommand,
)
from geometry_msgs.msg import TransformStamped
from collections import deque

from geometry_msgs.msg import Pose as pose_msg


class TFInterface:
    def __init__(self):
        self.tfBuffer = tf2_ros.Buffer()  # Using default cache time of 10 secs
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.broadcaster = tf2_ros.TransformBroadcaster()
        time.sleep(1.0)

    def updateTF(self, source_frame, target_frame, pose):
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

    def get_frame_to_frame_transform(
        self, camera_info_data, frame_A="base_link", target_frame="camera_color_optical_frame"
    ):
        stamp = camera_info_data.header.stamp
        try:
            transform = self.tfBuffer.lookup_transform(
                frame_A,
                target_frame,
                rospy.Time(secs=stamp.secs, nsecs=stamp.nsecs),
            )
            return transform
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            return None

    def make_homogeneous_transform(self, transform):
        A_to_B = np.zeros((4, 4))
        A_to_B[:3, :3] = Rotation.from_quat(
            [
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w,
            ]
        ).as_matrix()
        A_to_B[:3, 3] = np.array(
            [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ]
        ).reshape(1, 3)
        A_to_B[3, 3] = 1
        return A_to_B

    def pose_to_matrix(self, pose):
        position = pose[0]
        orientation = pose[1]
        pose_matrix = np.zeros((4, 4))
        pose_matrix[:3, 3] = position
        pose_matrix[:3, :3] = Rotation.from_quat(orientation).as_matrix()
        pose_matrix[3, 3] = 1
        return pose_matrix

    def matrix_to_pose(self, mat):
        position = mat[:3, 3]
        orientation = Rotation.from_matrix(mat[:3, :3]).as_quat()
        return (position, orientation)


class ArUcoPerception(TFInterface):
    def __init__(self, num_perception_samples=25):
        """
        We track separate queues for each marker ID we care about.
        Also set up multiple publishers:
         - A single "generic" publisher: /aruco_pose
         - One publisher per marker ID: /aruco_pose_0, /aruco_pose_1
        """
        self.num_perception_samples = num_perception_samples
        self.pose_queues = {
            0: deque(maxlen=num_perception_samples),
            1: deque(maxlen=num_perception_samples),
        }

        self.bridge = CvBridge()

        # Existing (generic) publisher
        self.aruco_pose_publisher = rospy.Publisher("/aruco_pose", Pose, queue_size=10)

        # New, marker-specific publishers
        self.aruco_pose_publisher_0 = rospy.Publisher("/aruco_pose_0", Pose, queue_size=10)
        self.aruco_pose_publisher_1 = rospy.Publisher("/aruco_pose_1", Pose, queue_size=10)

        # Subscribe to color and depth + camera info
        self.color_image_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
        self.camera_info_sub = message_filters.Subscriber("/camera/color/camera_info", CameraInfo)
        self.depth_image_sub = message_filters.Subscriber(
            "/camera/aligned_depth_to_color/image_raw", Image
        )
        ts = message_filters.TimeSynchronizer(
            [self.color_image_sub, self.camera_info_sub, self.depth_image_sub], 1
        )
        ts.registerCallback(self.rgbdCallback)

        super().__init__()

    def rgbdCallback(self, rgb_image_msg, camera_info_msg, depth_image_msg):
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_image_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_image_msg, "32FC1")
        except CvBridgeError as e:
            print(e)
            return

        # Detect ArUco markers
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, rejected = detector.detectMarkers(rgb_image)

        if len(corners) == 0:
            return
        
        ids = ids.flatten()
        transform_camera_to_base = self.get_frame_to_frame_transform(camera_info_msg)
        if transform_camera_to_base is None:
            return

        base_to_camera = self.make_homogeneous_transform(transform_camera_to_base)

        # For each marker detected, check its ID and compute pose
        for (markerCorner, markerID) in zip(corners, ids):
            # Only handle certain IDs
            if markerID not in [0, 1]:
                continue

            # Each markerCorner is shape (1,4,2); reshape to (4,2)
            marker_corners = markerCorner.reshape((4, 2))

            # Our known 3D layout of the corners in marker coords
            # (assuming 8cm side length, each corner is +/- 0.04 along x,y)
            landmarks_model = np.array(
                [
                    [-0.04,  0.04, 0],
                    [ 0.04,  0.04, 0],
                    [ 0.04, -0.04, 0],
                    [-0.04, -0.04, 0],
                ]
            )

            # Convert corners from pixel to 3D (camera frame)
            valid_landmarks_model = []
            valid_landmarks_world = []
            for i in range(marker_corners.shape[0]):
                u, v = marker_corners[i, 0], marker_corners[i, 1]
                valid, pt_world = self.pixel2World(
                    camera_info_msg, int(u), int(v), depth_image
                )
                if valid:
                    valid_landmarks_model.append(landmarks_model[i])
                    valid_landmarks_world.append(pt_world)

            if len(valid_landmarks_world) < 4:
                # Not enough corners to fit the model for this marker
                continue

            valid_landmarks_model = np.array(valid_landmarks_model)
            valid_landmarks_world = np.array(valid_landmarks_world)

            # We do a Kabsch-Umeyama to find rotation + translation
            scale_fixed = 1.0
            s, R_est, t_est = self.kabschUmeyama(
                valid_landmarks_world, valid_landmarks_model, scale_fixed
            )

            # The average of the 3D corners in camera coords
            tag_pos_camframe = np.mean(valid_landmarks_world, axis=0)
            camera_to_tag = np.eye(4)
            camera_to_tag[:3, :3] = R_est
            camera_to_tag[:3, 3] = tag_pos_camframe

            # Then transform that to the base frame
            base_to_tag = base_to_camera @ camera_to_tag

            # Broadcast a TF named AR_tag_<markerID>
            self.updateTF("base_link", f"AR_tag_{markerID}", base_to_tag)

            # Update the running average pose for this marker, then publish
            self.update_aruco_pose(base_to_tag, markerID)

    def update_aruco_pose(self, aruco_pose_mat, markerID):
        """Store and publish the running-averaged pose for each marker."""
        pose_tuple = self.matrix_to_pose(aruco_pose_mat)
        self.pose_queues[markerID].append(pose_tuple)

        # Wait until we have at least num_perception_samples for that marker
        if len(self.pose_queues[markerID]) < self.num_perception_samples:
            return

        # Compute running-average position/orientation
        positions = [p[0] for p in self.pose_queues[markerID]]
        orientations = [p[1] for p in self.pose_queues[markerID]]

        avg_pos = np.mean(positions, axis=0)
        avg_ori = np.mean(orientations, axis=0)

        p = Pose()
        p.position.x = avg_pos[0]
        p.position.y = avg_pos[1]
        p.position.z = avg_pos[2]
        p.orientation.x = avg_ori[0]
        p.orientation.y = avg_ori[1]
        p.orientation.z = avg_ori[2]
        p.orientation.w = avg_ori[3]

        # 1) Publish to the existing (generic) publisher
        self.aruco_pose_publisher.publish(p)

        # 2) Publish to the marker‑ID‑specific publisher
        if markerID == 0:
            self.aruco_pose_publisher_0.publish(p)
        elif markerID == 1:
            self.aruco_pose_publisher_1.publish(p)

    def pixel2World(self, camera_info, image_x, image_y, depth_image):

        # print("(image_y,image_x): ",image_y,image_x)
        # print("depth image: ", depth_image.shape[0], depth_image.shape[1])

        if image_y >= depth_image.shape[0] or image_x >= depth_image.shape[1]:
            return False, None

        depth = depth_image[image_y, image_x]

        if math.isnan(depth) or depth < 0.05 or depth > 1.0:

            depth = []
            for i in range(-2,2):
                for j in range(-2,2):
                    if image_y+i >= depth_image.shape[0] or image_x+j >= depth_image.shape[1]:
                        return False, None
                    pixel_depth = depth_image[image_y+i, image_x+j]
                    if not (math.isnan(pixel_depth) or pixel_depth < 50 or pixel_depth > 1000):
                        depth += [pixel_depth]

            if len(depth) == 0:
                return False, None

            depth = np.mean(np.array(depth))

        depth = depth/1000.0 # Convert from mm to m

        fx = camera_info.K[0]
        fy = camera_info.K[4]
        cx = camera_info.K[2]
        cy = camera_info.K[5]  

        # Convert to world space
        world_x = (depth / fx) * (image_x - cx)
        world_y = (depth / fy) * (image_y - cy)
        world_z = depth

        return True, (world_x, world_y, world_z)

    def kabschUmeyama(self, A, B, scale):

        assert A.shape == B.shape

        # print("A Shape:", A.shape)
        # print("B Shape:", B.shape)

        # Calculate scaled B
        scaled_B = B*scale

        # Calculate translation using centroids
        A_centered = A - np.mean(A, axis=0)
        B_centered = scaled_B - np.mean(scaled_B, axis=0)

        # Calculate rotation using scipy

        R, rmsd = Rotation.align_vectors(A_centered, B_centered)

        # print("R: ",R.as_matrix().shape)
        # print("Scaled B: ",np.mean(scaled_B, axis=0).shape)

        t = np.mean(A, axis=0) - R.as_matrix()@np.mean(scaled_B, axis=0)

        return scale, R.as_matrix(), t


if __name__ == "__main__":
    rospy.init_node("ArUcoPerception")
    aruco_perception = ArUcoPerception()
    rospy.spin()
