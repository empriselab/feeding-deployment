"""Stream the static objects from the simulated scene description to RViz."""

import rospy
from visualization_msgs.msg import Marker
from feeding_deployment.simulation.scene_description import SceneDescription
from pybullet_helpers.geometry import Pose

# TODO: switch to create_scene_description_from_config here in and in
# collision detector?


def _add_cube(pose: Pose, half_extents: tuple[float, float, float],
              rgba: tuple[float, float, float, float],
              marker_id: int,
              publisher: rospy.Publisher,
              frame_id="sim/base_link") -> None:
    
    marker = Marker()

    marker.ns = "cube"
    marker.id = marker_id
    marker.type = Marker.CUBE
    marker.header.stamp = rospy.Time.now()
    marker.header.frame_id = frame_id
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

    publisher.publish(marker)


def visualize_static_scene_in_rviz(scene_description: SceneDescription):
    """Stream the static objects from the simulated scene description to RViz."""
    
    rospy.init_node("static_scene_description_visualizer", anonymous=True)
    marker_pub = rospy.Publisher("visualization_marker", Marker, queue_size=10)

    # Wait for RViz to subscribe to the topic.
    rospy.sleep(1)

    # Visualize the table.
    _add_cube(scene_description.table_pose,
              scene_description.table_half_extents,
              scene_description.table_rgba,
              marker_id=0,
              publisher=marker_pub)

    # Visualize the vention stand.
    _add_cube(scene_description.robot_holder_pose,
              scene_description.robot_holder_half_extents,
              scene_description.robot_holder_rgba,
              marker_id=1,
              publisher=marker_pub)

    # Visualize the conservative bounding box.
    _add_cube(scene_description.conservative_bb_pose,
              scene_description.conservative_bb_half_extents,
              scene_description.conservative_bb_rgba,
              marker_id=2,
              publisher=marker_pub)

    rospy.spin()

    

if __name__ == "__main__":
    scene_description = SceneDescription()

    try:
        visualize_static_scene_in_rviz(scene_description)
    except rospy.ROSInterruptException:
        pass
