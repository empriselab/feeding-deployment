"""An interface for perception (robot joints, human head poses, etc.)."""

import threading
import time

import numpy as np
from pybullet_helpers.geometry import Pose
from scipy.spatial.transform import Rotation as R

try:
    import rospy
    from sensor_msgs.msg import JointState

    from feeding_deployment.head_perception.ros_wrapper import HeadPerceptionROSWrapper
except ModuleNotFoundError:
    pass

from feeding_deployment.robot_controller.arm_client import Arm


def publish_joint_states(arm):

    # publish joint states
    joint_states_pub = rospy.Publisher("/robot_joint_states", JointState, queue_size=10)

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

    def __init__(self, robot_interface: Arm | None) -> None:
        self._robot_interface = robot_interface

        try:
            rospy.init_node("perception_interface", anonymous=True)

        except Exception:
            pass

        # publish joint states in separate thread
        if robot_interface is not None:
            joint_state_thread = threading.Thread(
                target=publish_joint_states, args=(self._robot_interface,)
            )
            joint_state_thread.start()

        # run head perception
        if robot_interface is None:
            self._head_perception = None
        else:
            self._head_perception = HeadPerceptionROSWrapper()
            # warm start head perception
            for _ in range(10):
                self._head_perception.run_head_perception()

    def get_robot_joints(self) -> "JointState":
        """Get the current robot joint state."""
        q, gripper_position = self._robot_interface.get_state()
        joint_state = q.tolist() + [gripper_position, gripper_position]
        return joint_state
    
    def get_camera_data(self): # Rajat ToDo: Add return type
        return self._head_perception.get_top_camera_data()  

    def get_head_perception_forque_target_pose(self) -> Pose:
        """Get a target of the forque from head perception."""
        if self._head_perception is not None:
            forque_target_transform = self._head_perception.run_head_perception()
        else:
            # Use a sensible default value for testing in simulation.
            forque_target_transform = np.array(
                [
                    [
                        0.05720315,
                        -0.00795624,
                        -0.99833086,
                        0.02325958,
                    ],
                    [-0.9842066, 0.16734664, -0.05772752, 0.5556016],
                    [0.16752662, 0.98586602, 0.00174218, 0.5478612],
                    [0.0, 0.0, 0.0, 1.0],
                ]
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
