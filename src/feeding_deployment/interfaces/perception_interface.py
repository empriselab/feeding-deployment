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


    from feeding_deployment.head_perception.ros_wrapper import HeadPerceptionROSWrapper
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
        else:
            # self._head_perception = None
            self._head_perception = HeadPerceptionROSWrapper(record_goal_pose)
            
            # warm start head perception
            self._head_perception.set_tool("fork")
            for _ in range(10):
                self._head_perception.run_head_perception()

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

        # Rajat ToDo: Update this to return Pose type
        fork_tip_staging_pose = np.eye(4)
        fork_tip_staging_pose[:3, 3] = [0.250, 0.272, 0.518]
        fork_tip_staging_pose[:3, :3] = R.from_quat([0.523, -0.503, -0.469, 0.503]).as_matrix()

        return fork_tip_staging_pose
    
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

