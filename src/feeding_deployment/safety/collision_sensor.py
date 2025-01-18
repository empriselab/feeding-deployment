"""
This senses a collision in the robot arm using joint force torque sensing
"""
try:
    import rospy
    from std_msgs.msg import Bool
    from sensor_msgs.msg import JointState
    ROSPY_IMPORTED = True
except ModuleNotFoundError:
    ROSPY_IMPORTED = False

import os
import math
import numpy as np
import argparse
from pathlib import Path
import pinocchio as pin
from pybullet_helpers.joint import JointPositions, JointVelocities

class CollisionSensor:
    """See docstring above."""

    # Threshold for collision detection
    COLLISION_THRESHOLD = 20.0

    def __init__(self):

        assert ROSPY_IMPORTED, "ROS is required for CollisionSensor"

        self._print_once = True
        self.file_path = Path(__file__).parent.parent / "control" / "robot_controller" / "urdfs"
        self.file_path = str(self.file_path)
        self.model = pin.buildModelFromUrdf(
            os.path.join(self.file_path, "gen3_robotiq_2f_85.urdf")
        )
        self.data = self.model.createData()
        self.q_pin = np.zeros(self.model.nq)

        self._collision_pub = rospy.Publisher("/collision_free", Bool, queue_size=1)
        self._joint_state_sub = rospy.Subscriber(
            "/robot_joint_states", JointState, self._joint_state_callback
        )

        self._disable_collision_sensor = False
        self._disable_collision_sensor_sub = rospy.Subscriber(
            "/disable_collision_sensor", Bool, self._disable_collision_sensor_callback
        )

    def _disable_collision_sensor_callback(self, msg: "Bool") -> None:
        if msg.data:
            print("Collision sensor disabled")
        else:
            print("Collision sensor enabled")
        self._disable_collision_sensor = msg.data

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
        joint_pos = list(joint_state_msg.position)
        joint_vel = list(joint_state_msg.velocity)
        joint_tau = list(joint_state_msg.effort)
        assert len(joint_pos) == 8, "Expected 8 joint positions"
        assert len(joint_vel) == 8, "Expected 8 joint velocities"
        assert len(joint_tau) == 8, "Expected 8 joint torques"
        
        if not self._disable_collision_sensor:
            has_collision = self.sense_collisions(joint_pos, joint_vel, joint_tau)
            self._collision_pub.publish(Bool(data=not has_collision))
            if has_collision and self._print_once:
                print("Collision detected by CollisionSensor")
                self._print_once = False
        else:
            self._collision_pub.publish(Bool(data=True))

    # Todo: Add JointTorques to pybullet_helpers.joint 
    def sense_collisions(self, joint_positions: JointPositions, joint_velocities: JointVelocities, torque_readings: JointPositions) -> bool:
        """Senses collisions with environment."""
        
        # print("joint_positions: ", joint_positions)
        # print("joint_velocities: ", joint_velocities)
        # print("torque_readings: ", torque_readings)

        if np.linalg.norm(torque_readings) < 1e-6:
            # No torque readings means joint compliant mode is on
            return False

        # Pinocchio joint configuration
        q_pin = np.array(
            [
                math.cos(joint_positions[0]),
                math.sin(joint_positions[0]),
                joint_positions[1],
                math.cos(joint_positions[2]),
                math.sin(joint_positions[2]),
                joint_positions[3],
                math.cos(joint_positions[4]),
                math.sin(joint_positions[4]),
                joint_positions[5],
                math.cos(joint_positions[6]),
                math.sin(joint_positions[6]),
            ]
        )
        dq = np.array(joint_velocities[:7])
        torque_reading = np.array(torque_readings[:7])

        gravity = pin.computeGeneralizedGravity(self.model, self.data, q_pin)
        torque_model = pin.rnea(self.model, self.data, q_pin, dq, np.zeros(7))

        error = np.abs(torque_reading - torque_model)
        max_error = np.max(error)

        # Very high error means collision, otherwise model error
        if max_error > self.COLLISION_THRESHOLD:
            return True
        
        return False
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    rospy.init_node("collision_sensor")
    monitor = CollisionSensor()
    rospy.spin()