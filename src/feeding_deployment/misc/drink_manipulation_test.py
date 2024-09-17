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
from geometry_msgs.msg import Point
from visualization_msgs.msg import MarkerArray, Marker

from feeding_deployment.robot_controller.arm_client import ArmInterfaceClient
from feeding_deployment.robot_controller.command_interface import CartesianCommand
from geometry_msgs.msg import TransformStamped

# from feeding_deployment.head_perception.ros_wrapper import HeadPerceptionROSWrapper

from geometry_msgs.msg import Pose as pose_msg

class ArUcoPerception:
    def __init__(self):
        rospy.init_node('ArUcoPerception')
        self.AR_center_pose = None # get rid of this later

        self.bridge = CvBridge()

        self.cartesian_state_sub = message_filters.Subscriber('/robot_cartesian_state', pose_msg)
        self.cartesian_state_sub.registerCallback(self.follow_AR_tag)

        self.color_image_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        self.camera_info_sub = message_filters.Subscriber('/camera/color/camera_info', CameraInfo)
        self.depth_image_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        ts = message_filters.TimeSynchronizer([self.color_image_sub, self.camera_info_sub, self.depth_image_sub], 1)
        ts.registerCallback(self.rgbdCallback)

        self.voxel_publisher =  rospy.Publisher("/head_perception/voxels/marker_array", MarkerArray, queue_size=10)

        self.tfBuffer = tf2_ros.Buffer()  # Using default cache time of 10 secs
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.broadcaster = tf2_ros.TransformBroadcaster()
        time.sleep(1.0)

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

        transform = self.get_base_to_camera_transform(camera_info_msg)

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








        self.visualizeVoxels(landmarks_model_camera_frame)

    def visualizeVoxels(self, voxels):

        # print(voxels)

        markerArray = MarkerArray()

        marker = Marker()
        marker.header.seq = 0
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = "camera_color_optical_frame"
        marker.ns = "visualize_voxels"
        marker.id =  1
        marker.type = 6; # CUBE LIST
        marker.action = 0; # ADD
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
            point.x = voxels[i,0]
            point.y = voxels[i,1]
            point.z = voxels[i,2]

            marker.points.append(point)


        # average of the voxels is the center of the AR tag
        center = np.mean(voxels, axis=0)
        point = Point()
        point.x = center[0]
        point.y = center[1]
        point.z = center[2]
        marker.points.append(point)
        # 
        markerArray.markers.append(marker)
        self.voxel_publisher.publish(markerArray)

        return center

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
    
    def get_base_to_camera_transform(self, camera_info_data):
        target_frame = "camera_color_optical_frame"
        stamp = camera_info_data.header.stamp
        try:
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

        print(t)

        self.broadcaster.sendTransform(t)


    def follow_AR_tag(self, cartesian_state_msg):


        if self.AR_center_pose is None: return

        tag_corner_pos, tag_rot = self.AR_center_pose
        tag_pos = np.mean(tag_corner_pos, axis=0)

        q = cartesian_state_msg.orientation
        q = q.x, q.y, q.z, q.w
        robot_rot = Rotation.from_quat(q).as_matrix()
        robot_pos = cartesian_state_msg.position
        robot_pos = np.array([robot_pos.x, robot_pos.y, robot_pos.z])
        

        # print(robot_pos)

        print(tag_pos)

        # distance_to_cup = np.linalg.norm(np.array(robot_pos) - np.array(self.AR_center_pose))
        # print(distance_to_cup)


        # [ 0.23372017 -0.11247288  0.58964063]
        # [-0.14779702 -0.12871058  0.58610937]



        





        # print(robot_rot)

        # tag_pos, tag_rot = self.AR_center_pose
        # q = Rotation.from_matrix(AR_rot).as_quat()



        # print(f"AR_pos: {AR_pos}, AR_rot: {q}") # AR_pos: [-0.07119214 -0.0782794   0.51123438], AR_rot: [0.71363607 0.65925799 0.03252051 0.23461644]
        

        # print(robot_tool_pose)
        

if __name__ == '__main__':



    # target_position = (0.61, -0.12, 0.6)
    # target_orientation = (0.56, 0.37, 0.49, 0.55)
    # cmd = CartesianCommand(target_position, target_orientation)
    # robot_interface = ArmInterfaceClient()
    # robot_interface.execute_command(cmd)

    aruco_perception = ArUcoPerception()
    rospy.spin()
