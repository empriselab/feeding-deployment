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

    def __init__(self, robot_interface: ArmInterfaceClient | None, record_goal_pose: bool = False) -> None:
        self._robot_interface = robot_interface

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

    def get_head_perception_forque_target_pose(self, simulation = False) -> Pose:
        """Get a target of the forque from head perception."""
        if self._head_perception is not None and not simulation:
            forque_target_transform = self._head_perception.run_head_perception()
            print("\n--\n---\n----Forque target transform: ", forque_target_transform)
        else:
            # Use a sensible default value for testing in simulation.
            forque_target_transform = np.array(
                [[ 2.39288367e-02,  8.46555150e-04, -9.99713306e-01, -9.36197722e-02],
                [-9.98958576e-01, -3.88389663e-02, -2.39436604e-02,  4.75341624e-01],
                [-3.88481010e-02,  9.99245124e-01, -8.36977532e-05,  6.02467578e-01],
                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
            )
        forque_target_pose = Pose(
            (
                forque_target_transform[0, 3],
                forque_target_transform[1, 3],
                forque_target_transform[2, 3],
            ),
            R.from_matrix(forque_target_transform[:3, :3]).as_quat(),
        )
        return forque_target_pose
    
    def wait_for_user_continue_button(self) -> None:
        print("Waiting for transfer complete button press / ft sensor trigger ...")
        msg = rospy.wait_for_message("/transfer_complete", Bool)
        assert msg.data
        print("Received message, continuing ...")

