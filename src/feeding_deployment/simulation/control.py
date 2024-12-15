"""Functions for robot control using the feeding deployment simulator."""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import time
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from pybullet_helpers.geometry import Pose
from pybullet_helpers.gui import visualize_pose

from feeding_deployment.simulation.simulator import FeedingDeploymentPyBulletSimulator
from feeding_deployment.simulation.state import FeedingDeploymentSimulatorState

# Rajat ToDo: Use a single yaml file to store all the constants across sim and real robot
DAMPING_FACTOR = 0.05
DISTANCE_LOOKAHEAD = 0.04
ANGULAR_LOOKAHEAD = 5*np.pi/180
TIMESTEP = 1/240 # Default timestep in pybullet

def _get_trajectory_to_pose(
    target_pose: Pose,
    sim: FeedingDeploymentPyBulletSimulator,
    max_control_time: float,
    exclude_collision_ids: set[int] | None = None,
) -> list[FeedingDeploymentSimulatorState]:
    """
    Returns a joint trajectory to move the robot to a target pose as the actual robot controller would.
    """

    # visualize the target pose
    visualize_pose(target_pose, sim.physics_client_id)
    visualize_pose(sim.robot.get_end_effector_pose(), sim.physics_client_id)
    input("Press Enter to continue...")
        
    # Rajat ToDo: make sure the controller is working

    sim_states: list[FeedingDeploymentSimulatorState] = []

    def compute_next_step(sim, target_pose):
        current_pose = sim.robot.get_end_effector_pose()

        source_position = np.array(current_pose.position)
        source_orientation = R.from_quat(current_pose.orientation)
        target_position = np.array(target_pose.position)
        target_orientation = R.from_quat(target_pose.orientation)

        position_error = np.linalg.norm(source_position - target_position)
        orientation_error = np.linalg.norm(R.from_matrix(np.dot(source_orientation.as_matrix(), target_orientation.as_matrix().T)).as_rotvec())

        if position_error <= DISTANCE_LOOKAHEAD:
            target_waypoint_position = target_position
        else:
            target_waypoint_position = source_position + DISTANCE_LOOKAHEAD*(target_position - source_position)/position_error

        if orientation_error <= ANGULAR_LOOKAHEAD:
            target_waypoint_orientation = target_orientation.as_quat()
        else:
            key_times = [0, 1]
            key_rots = R.concatenate((source_orientation, target_orientation))
            slerp = Slerp(key_times, key_rots)

            interp_rotations = slerp([ANGULAR_LOOKAHEAD/orientation_error]) #second last is also aligned
            target_waypoint_orientation = interp_rotations[0].as_quat()

        # visualize_pose(current_pose, sim.physics_client_id)
        # visualize_pose(Pose(position=target_waypoint_position, orientation=target_waypoint_orientation), sim.physics_client_id)
        # input("Press Enter to continue...")

        n_dof = 7 # Rajat ToDo: Remove hardcoding

        # Use damped least squares to compute the joint velocities
        J = sim.robot.get_jacobian()
        # print("J.shape", J.shape)
        J = J[:, :n_dof]

        pos_error = source_position - target_waypoint_position

        # Convert to Rotation objects
        R_c = R.from_quat(current_pose.orientation)
        R_d = R.from_quat(target_waypoint_orientation)

        # Adjust quaternions to be on the same hemisphere
        if np.dot(R_d.as_quat(), R_c.as_quat()) < 0.0:
            R_c = R.from_quat(-R_c.as_quat())

        # Compute error rotation
        error_rotation = R_c.inv() * R_d

        # Convert error rotation to quaternion
        error_quat = error_rotation.as_quat()

        # Extract vector part
        orient_error_vector = error_quat[:3]

        # Get rotation matrix of nominal pose
        R_c_matrix = R_c.as_matrix()

        # Compute orientation error
        orient_error = -R_c_matrix @ orient_error_vector

        # Assemble error
        error = np.zeros(6)
        error[:3] = pos_error
        error[3:] = orient_error 

        damping_lambda = DAMPING_FACTOR * np.eye(n_dof)
        J_JT = J.T @ J + damping_lambda
                
        # J_damped = np.linalg.inv(J_JT) @ J.T

        # this is faster than the above commented computation
        c, lower = cho_factor(J_JT)
        J_n_damped = cho_solve((c, lower), J.T)

        joint_velocities = -J_n_damped @ error

        target_positions = sim.robot.get_joint_positions()[:n_dof] + joint_velocities * TIMESTEP

        return target_positions
        
    start_time = time.time()
    # while time.time() - start_time < max_control_time:
    while True:
        if target_pose.allclose(sim.robot.get_end_effector_pose()):
            break
        target_positions = compute_next_step(sim, target_pose)
        target_positions = np.concatenate((target_positions, [0, 0, 0, 0, 0, 0])) # Rajat ToDo: Remove hardcoding
        sim.set_motors(target_positions)

    return sim_states