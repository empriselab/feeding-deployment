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
    from geometry_msgs.msg import TransformStamped, Pose as PoseMsg
    from feeding_deployment.head_perception.ros_wrapper import HeadPerceptionROSWrapper
    ROSPY_IMPORTED = True
except ModuleNotFoundError as e:
    print(f"ROS not imported: {e}")
    ROSPY_IMPORTED = False

from feeding_deployment.robot_controller.arm_client import ArmInterfaceClient
from feeding_deployment.simulation.scene_description import SceneDescription

class RVizInterface:
    """An interface for visualization on rviz"""

    def __init__(self, scene_description: SceneDescription) -> None:

        assert ROSPY_IMPORTED, "ROS is required to run RVizInterface"

        self._scene_description = scene_description

        # Create publishers for rviz simulation.
        self.sim_joint_publishers = rospy.Publisher("/sim/robot_joint_states", JointState, queue_size=10)
        self.marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size=10)
        self.utensil_visualization_pub = rospy.Publisher('utensil_visualization_marker_array', MarkerArray, queue_size=10)
        self.food_visualization_pub = rospy.Publisher('food_visualization_marker_array', MarkerArray, queue_size=10)
        
        # Create a static transform broadcaster for rviz simulation.
        self.static_transform_broadcaster = tf2_ros.StaticTransformBroadcaster()

        # Create a broadcaster for tf2 transforms.
        self.broadcaster = tf2_ros.TransformBroadcaster()

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
        self.tool_update(False, "drink", self._scene_description.drink_pose)
        self.tool_update(False, "wipe", self._scene_description.wipe_pose)
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
        elif held_object == "drink":
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

    def publishTransformationToTF(self, source_frame, target_frame, transform):

        t = TransformStamped()

        t.header.stamp = rospy.Time.now()
        t.header.frame_id = source_frame
        t.child_frame_id = target_frame

        t.transform.translation.x = transform[0][3]
        t.transform.translation.y = transform[1][3]
        t.transform.translation.z = transform[2][3]

        rot = R.from_matrix(transform[:3,:3]).as_quat()
        t.transform.rotation.x = rot[0]
        t.transform.rotation.y = rot[1]
        t.transform.rotation.z = rot[2]
        t.transform.rotation.w = rot[3]

        self.broadcaster.sendTransform(t)

    def visualizeTransform(self, source_frame, target_frame, transform):

        self.publishTransformationToTF(source_frame, target_frame, transform)

    def visualize_plan(self, plan):
        for sim_state in plan:
            self.joint_state_update(sim_state.robot_joints)
            time.sleep(0.1)

    def get_pose_msg_from_transform(self, transform):

        pose = PoseMsg()
        pose.position.x = transform[0,3]
        pose.position.y = transform[1,3]
        pose.position.z = transform[2,3]

        quat = R.from_matrix(transform[:3,:3]).as_quat()
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]

        return pose

    def visualize_fork(self, transform):
        print("Visualizing fork")
        # visualize fork mesh in rviz
        marker_array = MarkerArray()
        marker = Marker()
        marker.id = 0
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = "base_link"
        marker.type = marker.MESH_RESOURCE
        marker.action = marker.ADD
        marker.scale.x = 0.001
        marker.scale.y = 0.001
        marker.scale.z = 0.001
        marker.color.a = 1.0
        
        # marker color is grey
        marker.color.r = 0.5
        marker.color.g = 0.5
        marker.color.b = 0.5
        
        pose = self.get_pose_msg_from_transform(transform)
        marker.pose = pose

        marker.mesh_resource = "package://kortex_description/tools/feeding_utensil/fork_tip.stl"
        marker_array.markers.append(marker)

        self.utensil_visualization_pub.publish(marker_array)

    def visualize_food(self, transform, id = 0):

        # publish a cube marker
        marker_array = MarkerArray()
        marker = Marker()
        marker.id = id
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = "base_link"
        marker.type = marker.CUBE
        marker.action = marker.ADD
        marker.scale.x = 0.01
        marker.scale.y = 0.01
        marker.scale.z = 0.01
        marker.color.a = 1.0

        # marker color is red
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        pose = self.get_pose_msg_from_transform(transform)
        marker.pose = pose

        marker_array.markers.append(marker)

        self.food_visualization_pub.publish(marker_array)

    