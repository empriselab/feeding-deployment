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
from pathlib import Path
from feeding_deployment.simulation.simulator import (
    FeedingDeploymentPyBulletSimulator,
)
from feeding_deployment.simulation.scene_description import SceneDescription, create_scene_description_from_config
from pybullet_helpers.inverse_kinematics import add_fingers_to_joint_positions, check_collisions_with_held_object
from pybullet_helpers.joint import JointPositions, JointVelocities


class CollisionMonitor:
    """See docstring above."""

    def __init__(self, scene_config: str, transfer_type: str, use_ros: bool = True):
        self._use_ros = use_ros
        scene_config_path = Path(__file__).parent.parent / "simulation" / "configs" / f"{scene_config}.yaml"
        self._scene_description = create_scene_description_from_config(scene_config_path, transfer_type)
        self._sim = FeedingDeploymentPyBulletSimulator(self._scene_description, use_gui=False, ignore_user=True)

        self._print_once = True

        if use_ros:
            assert ROSPY_IMPORTED, "rospy was not imported"
            self._collision_pub = rospy.Publisher("/collision_free", Bool, queue_size=1)
            self._within_joint_limits_pub = rospy.Publisher("/within_joint_limits", Bool, queue_size=1)
            self._joint_state_sub = rospy.Subscriber(
                "/robot_joint_states", JointState, self._joint_state_callback
            )

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
        # has_collision = self.check_collisions(combined_joint_state)
        has_collision = self.sense_collisions(combined_joint_state)
        self._collision_pub.publish(Bool(data=not has_collision))
        if has_collision and self._print_once:
            print("Collision detected by CollisionMonitor")
            self._print_once = False

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


    # Todo: Add JointTorques to pybullet_helpers.joint 
    def sense_collisions(self, joint_positions: JointPositions, joint_velocities: JointVelocities, torque_readings: JointPositions) -> bool:
        """Check collisions with all objects."""
        print("joint_positions: ", joint_positions)
        print("joint_velocities: ", joint_velocities)
        print("torque_readings: ", torque_readings)
        return False
        # return self._sim.check_collisions(joint_positions)
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_config", type=str, default="vention")
    parser.add_argument("--transfer_type", type=str, default="inside")
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()

    if args.dry:
        monitor = CollisionMonitor(scene_config=args.scene_config, transfer_type=args.transfer_type, use_ros=False)
        # A few tests.
        assert not monitor.check_collisions([2.8884768101246143, -0.7913320348241513, -1.7742571378056136, -2.078073911389284, 2.2868481461996795, -0.8264030187967055, -0.11233229012519357, 0.44, 0.44, 0.44, 0.44, -0.44, -0.44])
        assert monitor.check_collisions([2.0, -0.7913320348241513, -1.7742571378056136, -2.078073911389284, 2.2868481461996795, -0.8264030187967055, -0.11233229012519357, 0.44, 0.44, 0.44, 0.44, -0.44, -0.44])
    else:
        rospy.init_node("collision_free_monitor")
        monitor = CollisionMonitor(scene_config=args.scene_config, transfer_type=args.transfer_type)
        rospy.spin()
    
    


