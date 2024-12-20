"""Functions for robot control using the feeding deployment simulator."""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from pybullet_helpers.geometry import Pose
from pybullet_helpers.joint import JointPositions

# Rajat ToDo: Use a single yaml file to store all the constants across sim and real robot
DAMPING_FACTOR = 0.05
DISTANCE_LOOKAHEAD = 0.04
ANGULAR_LOOKAHEAD = 5*np.pi/180
TIMESTEP = 1/240 # Default timestep in pybullet

def cartesian_control_step(current_joint_positions: JointPositions, current_jacobian: np.ndarray, current_pose: Pose, target_pose: Pose) -> JointPositions:

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

    J = current_jacobian
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

    target_positions = current_joint_positions[:n_dof] + joint_velocities * TIMESTEP

    return target_positions