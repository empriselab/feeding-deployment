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
    from visualization_msgs.msg import MarkerArray, Marker
    import tf2_ros
    from geometry_msgs.msg import TransformStamped

    from feeding_deployment.head_perception.ros_wrapper import HeadPerceptionROSWrapper
except ModuleNotFoundError:
    pass

from feeding_deployment.robot_controller.arm_client import ArmInterfaceClient
from feeding_deployment.simulation.scene_description import SceneDescription

class RVizInterface:
    """An interface for visualization on rviz"""

    def __init__(self, scene_description: SceneDescription) -> None:

        self._scene_description = scene_description

        # Create publishers for rviz simulation.
        self.sim_joint_publishers = rospy.Publisher("/sim/robot_joint_states", JointState, queue_size=10)
        self.marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size=10)
        
        # Create a static transform broadcaster for rviz simulation.
        self.static_transform_broadcaster = tf2_ros.StaticTransformBroadcaster()

        # Wait for RViz to subscribe to the topic.
        rospy.sleep(1)

        # Visualize the table.     
        self._add_cube(self._scene_description.table_pose,
                self._scene_description.table_half_extents,
                self._scene_description.table_rgba,
                marker_id=0)

        # Visualize the vention stand.
        self._add_cube(self._scene_description.robot_holder_pose,
                self._scene_description.robot_holder_half_extents,
                self._scene_description.robot_holder_rgba,
                marker_id=1)

        # Visualize the conservative bounding box.
        self._add_cube(self._scene_description.conservative_bb_pose,
                self._scene_description.conservative_bb_half_extents,
                self._scene_description.conservative_bb_rgba,
                marker_id=2)

        # set initial joint states
        self.joint_state_update(self._scene_description.initial_joints)

        # Set initial tool states
        self.tool_update(False, "cup", self._scene_description.cup_pose)
        self.tool_update(False, "wiper", self._scene_description.wiper_pose)
        self.tool_update(False, "utensil", self._scene_description.utensil_pose)
    
    def joint_state_update(self, joints: JointPositions):
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
        
    def tool_update(self, pick: bool, held_object: str, object_pose: Pose) -> None:

        if held_object == "utensil":
            tool_base = "forkbase"
        elif held_object == "cup":
            tool_base = "drinkbase"
        elif held_object == "wiper":
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

    def _add_cube(self, 
                pose: Pose, half_extents: tuple[float, float, float],
                rgba: tuple[float, float, float, float],
                marker_id: int) -> None:
        
        marker = Marker()

        marker.ns = "cube"
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = "sim/base_link"
        marker.action = marker.ADD

        marker.pose.position.x = pose.position[0]
        marker.pose.position.y = pose.position[1]
        marker.pose.position.z = pose.position[2]
        marker.pose.orientation.x = pose.orientation[0]
        marker.pose.orientation.y = pose.orientation[1]
        marker.pose.orientation.z = pose.orientation[2]
        marker.pose.orientation.w = pose.orientation[3]

        marker.scale.x = 2 * half_extents[0]
        marker.scale.y = 2 * half_extents[1]
        marker.scale.z = 2 * half_extents[2]

        marker.color.r = rgba[0]
        marker.color.g = rgba[1]
        marker.color.b = rgba[2]
        marker.color.a = rgba[3]

        self.marker_pub.publish(marker)