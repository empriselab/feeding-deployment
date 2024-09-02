"""An interface for perception (robot joints, human head poses, etc.)."""

import threading
import time

import numpy as np
from pybullet_helpers.geometry import Pose
from pybullet_helpers.joint import JointPositions
from scipy.spatial.transform import Rotation as R


try:
    import rospy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String
    from visualization_msgs.msg import MarkerArray
    import tf2_ros
    from geometry_msgs.msg import TransformStamped

    from feeding_deployment.head_perception.ros_wrapper import HeadPerceptionROSWrapper
except ModuleNotFoundError:
    pass

from feeding_deployment.robot_controller.arm_client import ArmInterfaceClient

class PerceptionInterface:
    """An interface for perception (robot joints, human head poses, etc.)."""

    def __init__(self, robot_interface: ArmInterfaceClient | None) -> None:
        self._robot_interface = robot_interface
        
        # Create a shared publisher for rviz simulation.
        self.sim_joint_publishers = rospy.Publisher("/sim/robot_joint_states", JointState, queue_size=10)
        self.static_transform_broadcaster = tf2_ros.StaticTransformBroadcaster()

        # run head perception
        if robot_interface is None:
            self._head_perception = None
        else:
            self._head_perception = None
            # self._head_perception = HeadPerceptionROSWrapper()
            # # warm start head perception
            # for _ in range(10):
            #     self._head_perception.run_head_perception()

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
    
    def rviz_joint_state_update(self, joints: JointPositions):
        self.sim_joint_publishers.publish(
                JointState(
                    name=[
                        "joint_1", "joint_2", "joint_3", 
                        "joint_4", "joint_5", "joint_6", 
                        "joint_7", "finger_joint"
                    ],
                    position=joints[:7] + [0.0]  # Assuming you want to add 0.0 for the finger_joint
                )
            )
        
    def rviz_tool_update(self, pick: bool, held_object: str, object_pose: Pose) -> None:

        if held_object == "utensil":
            tool_base = "forkbase"
        elif held_object == "cup":
            tool_base = "drinkbase"
        elif held_object == "wipe":
            tool_base = "wipebase"

        if pick:
            self.publish_static_transform("sim/finger_tip", "sim/" + tool_base, object_pose)
        else:
            self.publish_static_transform("sim/base_link", "sim/" + tool_base, object_pose)
    
    def publish_static_transform(self, parent_frame: str, child_frame: str, pose: Pose) -> None:

        static_transform_stamped = TransformStamped()

        static_transform_stamped.header.stamp = rospy.Time.now()
        static_transform_stamped.header.frame_id = parent_frame
        static_transform_stamped.child_frame_id = child_frame

        static_transform_stamped.transform.translation.x = pose.position[0]
        static_transform_stamped.transform.translation.y = pose.position[1]
        static_transform_stamped.transform.translation.z = pose.position[2]

        static_transform_stamped.transform.rotation.x = pose.orientation[0]
        static_transform_stamped.transform.rotation.y = pose.orientation[1]
        static_transform_stamped.transform.rotation.z = pose.orientation[2]
        static_transform_stamped.transform.rotation.w = pose.orientation[3]

        self.static_transform_broadcaster.sendTransform(static_transform_stamped)
