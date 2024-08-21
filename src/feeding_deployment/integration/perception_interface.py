"""An interface for perception (robot joints, human head poses, etc.)."""

import threading
import time

import rospy
from pybullet_helpers.geometry import Pose, multiply_poses
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState

from feeding_deployment.head_perception.ros_wrapper import HeadPerceptionROSWrapper
from feeding_deployment.robot_controller.arm_client import Arm


def publish_joint_states(arm):

    # publish joint states
    joint_states_pub = rospy.Publisher("/joint_states", JointState, queue_size=10)

    while not rospy.is_shutdown():
        arm_pos, gripper_pos = arm.get_state()
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = rospy.Time.now()
        joint_state_msg.name = [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "joint_7",
            "finger_joint",
        ]
        joint_state_msg.position = arm_pos.tolist() + [gripper_pos]
        joint_state_msg.velocity = [0.0] * 8
        joint_state_msg.effort = [0.0] * 8
        joint_states_pub.publish(joint_state_msg)
        time.sleep(0.01)


class PerceptionInterface:
    """An interface for perception (robot joints, human head poses, etc.)."""

    def __init__(self, robot_interface: Arm) -> None:
        self._robot_interface = robot_interface

        # publish joint states in separate thread
        joint_state_thread = threading.Thread(
            target=publish_joint_states, args=(self._robot_interface,)
        )
        joint_state_thread.start()

        # run head perception
        self._head_perception = HeadPerceptionROSWrapper()

        # warm start head perception
        for _ in range(10):
            self._head_perception.run_head_perception()

    def get_robot_joints(self) -> JointState:
        """Get the current robot joint state."""
        q, gripper_position = self._robot_interface.get_state()
        joint_state = q.tolist() + [gripper_position, gripper_position]
        return joint_state

    def get_head_perception_forque_target_pose(self) -> Pose:
        """Get a target of the forque from head perception."""
        forque_target_transform = self._head_perception.run_head_perception()
        forque_target_pose = Pose(
            (
                forque_target_transform[0, 3],
                forque_target_transform[1, 3],
                forque_target_transform[2, 3],
            ),
            R.from_matrix(forque_target_transform[:3, :3]).as_quat(),
        )
        return forque_target_pose
