"""This creates a copy of the simulation environment, receives joint states
directly from the robot, and reports whether any collisions are detected.

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

import numpy as np
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
            self._within_joint_limits_pub = rospy.Publisher("/within_joint_limits", Bool, queue_size=1)
            self._joint_state_sub = rospy.Subscriber(
                "/robot_joint_states", JointState, self._joint_state_callback
            )
        self._use_ros = use_ros
        self._scene_description = SceneDescription()
        self._sim = FeedingDeploymentPyBulletSimulator(self._scene_description, use_gui=False, ignore_user=True)

        self.joint_limits = {
            "joint_2": {"lower": -2.24, "upper": 2.24},
            "joint_4": {"lower": -2.57, "upper": 2.57},
            "joint_6": {"lower": -2.09, "upper": 2.09},
        }

        # add padding of 10 degrees to the joint limits
        padding = 15 * np.pi / 180
        self.soft_joint_limits = {}
        for joint_name, limits in self.joint_limits.items():
            self.soft_joint_limits[joint_name] = {
                "lower": limits["lower"] + padding,
                "upper": limits["upper"] - padding,
            }

        padding = 15 * np.pi / 180
        self.hard_joint_limits = {}
        for joint_name, limits in self.joint_limits.items():
            self.hard_joint_limits[joint_name] = {
                "lower": limits["lower"] + padding,
                "upper": limits["upper"] - padding,
            }

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
        outside_soft_joint_limits = self.check_outside_soft_joint_limits(joint_state_msg.name, arm_joint_state)
        print("In Collision" if has_collision else "Collision Free")
        print("Outside Soft Joint Limits" if outside_soft_joint_limits else "Within Soft Joint Limits")
        self._collision_pub.publish(Bool(data=not (has_collision)))

        outside_hard_joint_limits = self.check_outside_hard_joint_limits(joint_state_msg.name, arm_joint_state)
        print("Outside Hard Joint Limits" if outside_hard_joint_limits else "Within Hard Joint Limits")
        self._within_joint_limits_pub.publish(Bool(data=not outside_hard_joint_limits))

    def check_outside_soft_joint_limits(self, joint_names: list[str], joint_positions: list[float]) -> bool:
        """Check if the joint positions are within the soft joint limits."""
        for joint_name, joint_position in zip(joint_names, joint_positions):
            if joint_name in self.soft_joint_limits:
                limits = self.soft_joint_limits[joint_name]
                if joint_position < limits["lower"] or joint_position > limits["upper"]:
                    return True
        return False

    def check_outside_hard_joint_limits(self, joint_names: list[str], joint_positions: list[float]) -> bool:
        """Check if the joint positions are within the hard joint limits."""
        for joint_name, joint_position in zip(joint_names, joint_positions):
            if joint_name in self.hard_joint_limits:
                limits = self.hard_joint_limits[joint_name]
                if joint_position < limits["lower"] or joint_position > limits["upper"]:
                    return True
        return False

    def check_collisions(self, joint_positions: JointPositions) -> bool:
        """Check collisions, but only with objects that can't be held."""
        collision_ids = self._sim.get_collision_ids()
        collision_ids -= {self._sim.drink_id, self._sim.utensil_id, self._sim.wipe_id}
        return check_collisions_with_held_object(
            self._sim.robot,
            collision_ids,
            self._sim.physics_client_id,
            held_object=None,
            base_link_to_held_obj=None,
            joint_state=joint_positions,
            distance_threshold=0.005, # 5mm
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
    
    


