"""This creates a copy of the simulation environment, receives joint states
directly from the robot, and reports whether any collisions are detected.

TODO: integrate with watchdog.

Note that possible collisions between held objects and world objects are not
checked because we cannot directly sense held object states from the robot.
"""
try:
    import rospy
    from std_msgs.msg import Bool
    from sensor_msgs.msg import JointState
    ROSPY_IMPORTED = True
except ModuleNotFoundError:
    ROSPY_IMPORTED = False

from feeding_deployment.simulation.simulator import (
    FeedingDeploymentPyBulletSimulator,
)
from feeding_deployment.simulation.scene_description import SceneDescription
from pybullet_helpers.inverse_kinematics import add_fingers_to_joint_positions, check_collisions_with_held_object
from pybullet_helpers.joint import JointPositions


class CollisionMonitor:
    """See docstring above."""

    def __init__(self, use_ros: bool = True):
        if use_ros:
            assert ROSPY_IMPORTED, "rospy was not imported"
            self._collision_pub = rospy.Publisher("/collision_free", Bool, queue_size=1)
            self._joint_state_sub = rospy.Subscriber(
                "/robot_joint_states", JointState, self._joint_state_callback
            )
        self._use_ros = use_ros
        self._scene_description = SceneDescription()
        self._sim = FeedingDeploymentPyBulletSimulator(self._scene_description)

    def _joint_state_callback(self, joint_state_msg: "JointState") -> None:
        # Convert joint state message into JointPositions.
        assert joint_state_msg.name == [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "joint_7",
            "finger_joint",
        ]
        assert len(joint_state_msg.position)
        joint_lst = list(joint_state_msg.position)
        assert len(joint_lst) == 8
        arm_joint_state, finger_state = joint_lst[:7], joint_lst[7]
        combined_joint_state = add_fingers_to_joint_positions(self._sim.robot,
                                                            arm_joint_state,
                                                            finger_state)
        # Run collision checking.
        has_collision = self.check_collisions(combined_joint_state)
        self._collision_pub.publish(Bool(data=has_collision))

    def check_collisions(self, joint_positions: JointPositions) -> bool:
        """Check collisions, but only with objects that can't be held."""
        collision_ids = self._sim.get_collision_ids()
        collision_ids -= {self._sim.cup_id, self._sim.utensil_id, self._sim.wiper_id}
        return check_collisions_with_held_object(
            self._sim.robot,
            collision_ids,
            self._sim.physics_client_id,
            held_object=None,
            base_link_to_held_obj=None,
            joint_state=joint_positions,
        )
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()

    if args.dry:
        monitor = CollisionMonitor(use_ros=False)
        # A few tests.
        assert not monitor.check_collisions([2.8884768101246143, -0.7913320348241513, -1.7742571378056136, -2.078073911389284, 2.2868481461996795, -0.8264030187967055, -0.11233229012519357, 0.44, 0.44, 0.44, 0.44, -0.44, -0.44])
        assert monitor.check_collisions([2.0, -0.7913320348241513, -1.7742571378056136, -2.078073911389284, 2.2868481461996795, -0.8264030187967055, -0.11233229012519357, 0.44, 0.44, 0.44, 0.44, -0.44, -0.44])
    else:
        rospy.init_node("collision_free_monitor")
        monitor = CollisionMonitor()
        rospy.spin()
    
    


