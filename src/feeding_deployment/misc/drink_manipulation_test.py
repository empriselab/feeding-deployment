# Description: This script is used to detect ArUco markers and estimate their pose in the camera frame.

# python imports
import os, sys
import cv2
import numpy as np
import time
import math
from scipy.spatial.transform import Rotation

# ros imports
import rospy
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import tf2_ros
from geometry_msgs.msg import Point, Pose
from visualization_msgs.msg import MarkerArray, Marker

from feeding_deployment.robot_controller.arm_client import ArmInterfaceClient
from feeding_deployment.robot_controller.command_interface import CartesianCommand, JointCommand, CloseGripperCommand
from geometry_msgs.msg import TransformStamped
from collections import deque

# from feeding_deployment.head_perception.ros_wrapper import HeadPerceptionROSWrapper

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

    def get_frame_to_frame_transform(self, camera_info_data, frame_A = "base_link", target_frame = "camera_color_optical_frame"):
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
            print("Exexption finding transform between base_link and", target_frame)
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
    def __init__(self):
        rospy.init_node('ArUcoPerception')

        self.bridge = CvBridge()
        self.aruco_pose_queue = deque(maxlen=10)
        self.aruco_pose_publisher =  rospy.Publisher("/aruco_pose", Pose, queue_size=10)

        self.color_image_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        self.camera_info_sub = message_filters.Subscriber('/camera/color/camera_info', CameraInfo)
        self.depth_image_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        ts = message_filters.TimeSynchronizer([self.color_image_sub, self.camera_info_sub, self.depth_image_sub], 1)
        ts.registerCallback(self.rgbdCallback)

        super().__init__()

    def rgbdCallback(self, rgb_image_msg, camera_info_msg, depth_image_msg):

        try:
            # Convert your ROS Image message to OpenCV2
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_image_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_image_msg, "32FC1")
        except CvBridgeError as e:
            print(e)

        arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        arucoParams = cv2.aruco.DetectorParameters_create()
        (corners, ids, rejected) = cv2.aruco.detectMarkers(rgb_image, arucoDict, parameters=arucoParams)

        if len(corners) > 0:
            ids = ids.flatten()
            for (markerCorner, markerID) in zip(corners, ids):
                corners = markerCorner.reshape((4, 2))
        else:
            return None  

        landmarks = corners
        landmarks_model = np.array([[-0.04,0.04,0],[0.04,0.04,0],[0.04,-0.04,0],[-0.04,-0.04,0]])

        # convert  2d landmarks to 3d world points 
        valid_landmarks_model = []
        valid_landmarks_world = []
        for i in range(landmarks.shape[0]):
            validity, point = self.pixel2World(camera_info_msg, landmarks[i,0].astype(int), landmarks[i,1].astype(int), depth_image)
            if validity:
                valid_landmarks_model.append(landmarks_model[i])
                valid_landmarks_world.append(point)

        if len(valid_landmarks_world) < 4:
            print("Not enough landmarks to fit model.")
            return

        valid_landmarks_model = np.array(valid_landmarks_model)
        valid_landmarks_world = np.array(valid_landmarks_world)

        scale_fixed = 1.0
        s, ret_R, ret_t = self.kabschUmeyama(valid_landmarks_world, valid_landmarks_model, scale_fixed)

        # print("landmarks_selected_model[:,:,np.newaxis].shape: ",landmarks_model[:,:,np.newaxis].shape)
        # print("ret_R.shape: ",ret_R.shape)
        landmarks_model_camera_frame = ret_t.reshape(3,1) + s * (ret_R @ landmarks_model[:,:,np.newaxis])
        landmarks_model_camera_frame = np.squeeze(landmarks_model_camera_frame)

        # get things into world frame

        tag_pos = np.append(np.mean(valid_landmarks_world, axis=0), 1) # pad with 1 for homogeneous coordinate

        transform = self.get_frame_to_frame_transform(camera_info_msg)

        if transform is not None:   
            base_to_camera = self.make_homogeneous_transform(transform)

            # cam to tag homogeneous transform
            camera_to_tag = np.zeros((4, 4))
            camera_to_tag[:3, :3] = ret_R
            camera_to_tag[:3, 3] = np.array([ tag_pos[0], tag_pos[1], tag_pos[2] ]).reshape(1, 3)
            camera_to_tag[3, 3] = 1 

            # base to tag homogeneous transform and update tf
            base_to_tag = np.dot(base_to_camera, camera_to_tag)
            self.updateTF("base_link", "AR_tag", base_to_tag)
            self.update_aruco_pose(base_to_tag)

    def update_aruco_pose(self, aruco_pose_mat):
        aruco_pose = self.matrix_to_pose(aruco_pose_mat)
        self.aruco_pose_queue.append(aruco_pose)
        running_average_position = np.mean([pose[0] for pose in self.aruco_pose_queue], axis=0)
        running_average_orientation = np.mean([pose[1] for pose in self.aruco_pose_queue], axis=0)
        p = Pose()
        p.position.x = running_average_position[0]
        p.position.y = running_average_position[1]
        p.position.z = running_average_position[2]
        p.orientation.x = running_average_orientation[0]
        p.orientation.y = running_average_orientation[1]
        p.orientation.z = running_average_orientation[2]
        p.orientation.w = running_average_orientation[3]
        self.aruco_pose_publisher.publish(p)

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


class DrinkManipulation(TFInterface):
    def __init__(self):
        self.robot_interface = ArmInterfaceClient()
        self.aruco_pose_sub =  message_filters.Subscriber("/aruco_pose", Pose)
        self.aruco_pose_sub.registerCallback(self.update_aruco_pose)
        self.aruco_pose = None
        super().__init__()

    def update_aruco_pose(self, msg):
        position = (msg.position.x, msg.position.y, msg.position.z)
        orientation = (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
        self.aruco_pose = (position, orientation)

    def get_pre_grasp_pose(self, aruco_pose):
        aruco_pos_mat = self.pose_to_matrix(aruco_pose)
        
        static_transform = np.zeros((4, 4))
        static_transform[:3, :3] = Rotation.from_euler("xyz", [np.pi, 0, np.pi / 2]).as_matrix()
        static_transform[:3, 3] = np.array([0.02, 0, 0.05])
        static_transform[3, 3] = 1 

        goal_frame = np.dot(aruco_pos_mat, static_transform)
        goal_pose = self.matrix_to_pose(goal_frame)
        return goal_pose

    def pick_up_drink(self):
        # Wait for the aruco pose to be found.
        while self.aruco_pose is None:
            time.sleep(0.1)

        # Close the fingers.
        self.close_fingers()

        # Move to the pre-grasp pose.
        pre_grasp_pose = self.get_pre_grasp_pose(self.aruco_pose)
        self.move_to_pose(pre_grasp_pose)
        

    def move_to_pose(self, pose):
        self.updateTF("base_link", "goal_frame", self.pose_to_matrix(pose))
        input("Press enter to go to goal pose")        

        cmd = CartesianCommand(pose[0], pose[1])
        self.robot_interface.execute_command(cmd)

    def close_fingers(self):
        self.robot_interface.execute_command(CloseGripperCommand())


if __name__ == '__main__':

    # target_position = (0.61, -0.12, 0.6)
    # target_orientation = (0.56, 0.37, 0.49, 0.55)
    # cmd = CartesianCommand(target_position, target_orientation)
    # robot_interface = ArmInterfaceClient()
    # robot_interface.execute_command(cmd)

    aruco_perception = ArUcoPerception()
    drink_manipulation = DrinkManipulation()

    # NOTE: these are 6dof-specific joint positions, would need to be updated
    input("Press enter to reset the robot to gaze position")
    gaze_joints = [1.125354671287698, 0.9558922716457215, -1.8328343338277167, -2.020429265225662, -1.428489781314294, -1.8853739747611273]
    drink_manipulation.robot_interface.execute_command(JointCommand(gaze_joints))

    input("Press enter to reset the robot to staging position")
    gaze_joints = [0.7834493286907556, 1.0428972118263922, -1.3236784134531359, -1.053118290932062, -1.9758625959637302, -2.263605433837996]
    drink_manipulation.robot_interface.execute_command(JointCommand(gaze_joints))

    input("Press enter to pick up the drink")
    drink_manipulation.pick_up_drink()

