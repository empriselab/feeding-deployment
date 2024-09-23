"""An interface for perception (robot joints, human head poses, etc.)."""

import threading
import time

import numpy as np
from pybullet_helpers.geometry import Pose
from pybullet_helpers.joint import JointPositions
from scipy.spatial.transform import Rotation as R
import json


try:
    import rospy
    from sensor_msgs.msg import JointState, CompressedImage
    from std_msgs.msg import String, Bool
    from visualization_msgs.msg import MarkerArray
    import tf2_ros
    from geometry_msgs.msg import TransformStamped
    from cv_bridge import CvBridge
    from geometry_msgs.msg import Pose as PoseMsg

    from feeding_deployment.head_perception.ros_wrapper import HeadPerceptionROSWrapper
    from feeding_deployment.aruco_perception.aruco_perception import ArUcoPerception
except ModuleNotFoundError:
    pass

from feeding_deployment.robot_controller.arm_client import ArmInterfaceClient

class PerceptionInterface:
    """An interface for perception (robot joints, human head poses, etc.)."""

    def __init__(self, robot_interface: ArmInterfaceClient | None, record_goal_pose: bool = False, simulate_head_perception: bool = False) -> None:
        self._robot_interface = robot_interface
        self._simulate_head_perception = simulate_head_perception

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        self.tool_tip_target_lock = threading.Lock()
        # this term is updated in the run_head_perception method and read in the get_tool_tip_pose method
        self.tool_tip_target_pose = None

        # run head perception
        if robot_interface is None:
            self._head_perception = None
            self._aruco_perception = None
        else:
            # self._head_perception = None
            self._head_perception = HeadPerceptionROSWrapper(record_goal_pose)
            
            # warm start head perception only if we're not recording the goal pose
            if not record_goal_pose:
                self._head_perception.set_tool("fork")
                for _ in range(10):
                    self._head_perception.run_head_perception()

            # Rajat ToDo: pass perception queues to all perception classes instead of having them use ros subscribers which spawn threads
            self._aruco_perception = ArUcoPerception()

        # Head perception thread setup
        self.head_perception_thread = None
        self.kill_the_thread = False
        self.head_perception_running = False
        
    def get_robot_joints(self) -> "JointState":
        """Get the current robot joint state."""
        joint_state_msg = rospy.wait_for_message("/robot_joint_states", JointState)
        q = np.array(joint_state_msg.position[:7])
        gripper_position = joint_state_msg.position[7]
        
        joint_state = q.tolist() + [
            gripper_position,
            gripper_position,
            gripper_position,
            gripper_position,
            -gripper_position,
            -gripper_position,
        ]
        return joint_state

    def get_camera_data(self):  # Rajat ToDo: Add return type
        return self._head_perception.get_camera_data()
    
    def set_head_perception_tool(self, tool: str) -> None:
        """Set the tool for head perception."""
        if self._head_perception is not None:
            self._head_perception.set_tool(tool)

    def start_head_perception_thread(self):
        assert not self.head_perception_running, "Head perception thread is already running"

        # Start head perception thread
        self.kill_the_thread = False
        self.head_perception_thread = threading.Thread(
            target=self.run_head_perception_thread, args=(), daemon=True
        )
        self.head_perception_thread.start()
        print("Head perception thread started")

    def run_head_perception_thread(self):
        self.head_perception_running = True

        t_init = time.time()
        while not self.kill_the_thread:
            t_now = time.time()
            step_time = t_now - t_init
            if step_time >= 0.02:  # 50 Hz
                if not self._simulate_head_perception:
                    tool_tip_target_pose = self._head_perception.run_head_perception()
                else:
                    tool_tip_target_pose = np.eye(4)
                    tool_tip_target_pose[:3, 3] = [-0.282, 0.540, 0.619]
                    tool_tip_target_pose[:3, :3] = R.from_quat([-0.490, 0.510, 0.511, -0.489]).as_matrix()
                with self.tool_tip_target_lock:
                    self.tool_tip_target_pose = tool_tip_target_pose
        self.head_perception_running = False

    def stop_head_perception_thread(self):
        if self.head_perception_running:
            self.kill_the_thread = True
            self.head_perception_thread.join()
            print("Head perception thread stopped")
        else:
            print("Head perception thread is not running")

    # Rajat ToDo: Change return type to Pose
    def get_head_perception_tool_tip_target_pose(self) -> np.ndarray:
        """Get a target of the forque from head perception."""
        with self.tool_tip_target_lock:
            return self.tool_tip_target_pose
        
    def get_tool_tip_pose(self) -> np.ndarray:

        arm_pos, ee_pose, gripper_pos = self._robot_interface.get_state()

        tool_tip_pose = np.eye(4)
        tool_tip_pose[:3, 3] = ee_pose[:3]
        tool_tip_pose[:3, :3] = R.from_quat(ee_pose[3:]).as_matrix()

        return tool_tip_pose
    
    def get_tool_tip_pose_at_staging(self) -> np.ndarray:

        tool_tip_staging_pose = np.eye(4)

        if self._head_perception.head_perception.tool == "fork":
            tool_tip_staging_pose[:3, 3] = [0.250, 0.272, 0.518]
            tool_tip_staging_pose[:3, :3] = R.from_quat([0.523, -0.503, -0.469, 0.503]).as_matrix()
        elif self._head_perception.head_perception.tool == "drink":
            tool_tip_staging_pose[:3, 3] = [0.229, 0.304, 0.609]
            tool_tip_staging_pose[:3, :3] = R.from_quat([0.523, -0.503, -0.469, 0.503]).as_matrix()
        elif self._head_perception.head_perception.tool == "wipe":
            tool_tip_staging_pose[:3, 3] = [0.417, 0.280, 0.509]
            tool_tip_staging_pose[:3, :3] = R.from_quat([0.523, -0.503, -0.469, 0.503]).as_matrix()

        return tool_tip_staging_pose
    
    def wait_for_user_continue_button(self) -> None:
        print("Waiting for transfer complete button press / ft sensor trigger ...")
        msg = rospy.wait_for_message("/transfer_complete", Bool)
        assert msg.data
        print("Received message, continuing ...")

    def getTransformationFromTF(self, source_frame, target_frame):

        while not rospy.is_shutdown():
            try:
                # print("Looking for transform")
                transform = self.tfBuffer.lookup_transform(source_frame, target_frame, rospy.Time())
                break
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                self.control_rate.sleep()
                continue

        T = np.zeros((4,4))
        T[:3,:3] = R.from_quat([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]).as_matrix()
        T[:3,3] = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]).reshape(1,3)
        T[3,3] = 1

        return T

    def perceive_drink_pickup_poses(self):
        
        # Rajat Hack: Wait one second for the aruco mean to be correct, does this actually help though?
        time.sleep(1)

        aruco_pose_msg = rospy.wait_for_message("/aruco_pose", PoseMsg)
        position = (aruco_pose_msg.position.x, aruco_pose_msg.position.y, aruco_pose_msg.position.z)
        orientation = (aruco_pose_msg.orientation.x, aruco_pose_msg.orientation.y, aruco_pose_msg.orientation.z, aruco_pose_msg.orientation.w)
        self.aruco_pose = (position, orientation)

        drink_poses  = {}

        drink_poses['pre_grasp_pose'] = self.get_aruco_relative_pose(self.get_pre_grasp_transform())
        drink_poses['inside_bottom_pose'] = self.get_aruco_relative_pose(self.get_inside_bottom_transform())
        drink_poses['inside_top_pose'] = self.get_aruco_relative_pose(self.get_inside_top_transform())
        drink_poses['post_grasp_pose'] = self.get_aruco_relative_pose(self.get_post_grasp_pose())
        drink_poses['place_inside_bottom_pose'] = self.get_aruco_relative_pose(self.get_place_inside_bottom_transform())
        drink_poses['place_pre_grasp_pose'] = self.get_aruco_relative_pose(self.get_place_pre_grasp_transform())

        self.last_drink_poses = drink_poses

        return drink_poses
    
    def record_drink_pickup_joint_pos(self):
        self.drink_pickup_joint_pos = self.get_robot_joints()[:7]

    def get_last_drink_pickup_configs(self):
        return self.last_drink_poses, self.drink_pickup_joint_pos

    def get_aruco_relative_pose(self, transform, override_angles = True):
        aruco_pos_mat = self.pose_to_matrix(self.aruco_pose)
        goal_frame = np.dot(aruco_pos_mat, transform)
        goal_pose = self.matrix_to_pose(goal_frame)

        # If true, use 2 hardcoded angle values.
        if override_angles:
            rot = R.from_quat(goal_pose[1])
            roll = np.pi / 2
            pitch = 0
            _, _, yaw = rot.as_euler("xyz")
            new_rot = R.from_euler("xyz", [roll, pitch, yaw])
            goal_pose = Pose(goal_pose[0], new_rot.as_quat())

        return goal_pose

    def get_pre_grasp_transform(self):
        tf = np.zeros((4, 4))
        tf[:3, :3] = R.from_euler("xyz", [np.pi, 0, np.pi / 2]).as_matrix()
        tf[:3, 3] = np.array([0.06, -0.02, 0.1]) # smaller x is more outside
        tf[3, 3] = 1
        return tf

    def get_inside_bottom_transform(self):
        tf = self.get_pre_grasp_transform()
        tf[2, 3] = 0.01
        return tf

    def get_inside_top_transform(self):
        tf = self.get_inside_bottom_transform()
        tf[0, 3] = 0.14
        return tf
    
    def get_post_grasp_pose(self):
        tf = self.get_inside_top_transform()
        tf[0, 3] = 0.2
        return tf
    
    def get_place_inside_bottom_transform(self):
        tf = self.get_inside_bottom_transform()
        # tf[1, 3] = 0.0
        return tf

    def get_place_pre_grasp_transform(self):
        tf = self.get_pre_grasp_transform()
        # tf[1, 3] = 0.0
        return tf

    def pose_to_matrix(self, pose):
        position = pose[0]
        orientation = pose[1]
        pose_matrix = np.zeros((4, 4))
        pose_matrix[:3, 3] = position
        pose_matrix[:3, :3] = R.from_quat(orientation).as_matrix()
        pose_matrix[3, 3] = 1
        return pose_matrix
    
    def matrix_to_pose(self, mat):
        position = mat[:3, 3]
        orientation = R.from_matrix(mat[:3, :3]).as_quat()
        return Pose(position, orientation) 