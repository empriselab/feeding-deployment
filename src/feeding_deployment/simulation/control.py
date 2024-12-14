"""Functions for robot control using the feeding deployment simulator."""

from __future__ import annotations

import logging
from typing import Callable

from pybullet_helpers.geometry import Pose
from pybullet_helpers.gui import visualize_pose

from feeding_deployment.simulation.simulator import FeedingDeploymentPyBulletSimulator
from feeding_deployment.simulation.state import FeedingDeploymentSimulatorState


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

    while True:
        pass
        
    # Rajat ToDo: make sure the controller is working

    sim_states: list[FeedingDeploymentSimulatorState] = []

    def compute_next_step(sim, sim_state, current_end_effector_pose, target_pose):
        # run a cartesian controller to get the next step
        target_positions = np.zeros(sim.robot.num_joints)
        return target_positions
        
    start_time = time.time()
    while time.time() - start_time < max_control_time:
        sim_state = sim.get_state()
        sim_states.append(sim_state)
        if sim.robot.get_end_effector_pose() == target_pose:
            break
        target_positions = compute_next_step(sim, sim_state)
        sim.set_motors(target_positions)

    return sim_states