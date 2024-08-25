"""Low-level actions that we can simulate and execute."""

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
from numpy.typing import NDArray
from copy import deepcopy, copy

from feeding_deployment.integration.perception_interface import PerceptionInterface
from feeding_deployment.integration.utils import simulated_trajectory_to_kinova_commands
from feeding_deployment.robot_controller.arm_client import Arm
from feeding_deployment.robot_controller.command_interface import KinovaCommand, CartesianCommand, JointCommand, OpenGripperCommand, CloseGripperCommand
from feeding_deployment.simulation.planning import (
    get_bite_transfer_plan,
    get_plan_to_grasp_cup,
    get_plan_to_grasp_utensil,
    get_plan_to_grasp_wiper,
    get_plan_to_stow_cup,
    get_plan_to_stow_utensil,
    get_plan_to_stow_wiper,
    get_plan_to_transfer_cup,
    get_plan_to_transfer_wiper,
    _get_motion_plan_for_robot_finger_tip,
    remap_trajectory_to_constant_distance,
    _get_plan_to_execute_grasp,
    _plan_to_sim_state_trajectory,
)
from pybullet_helpers.motion_planning import (
    run_motion_planning,
    try_direct_path,
    get_joint_positions_distance,
    run_smooth_motion_planning_to_pose,
    smoothly_follow_end_effector_path,
)
from pybullet_helpers.geometry import Pose
from feeding_deployment.simulation.simulator import FeedingDeploymentPyBulletSimulator
from feeding_deployment.simulation.state import FeedingDeploymentSimulatorState

def move_to_joint_positions(
    sim: FeedingDeploymentPyBulletSimulator,
    joint_positions: list[float],
) -> tuple[list[FeedingDeploymentSimulatorState], list[KinovaCommand]]:
    """Move the robot to the specified joint positions."""

    initial_joint_positions = sim.robot.get_joint_positions().copy()

    # Rajat ToDo: Check if I need to collsion check initial and target positions before calling this function
    direct_path = try_direct_path(robot = sim.robot,
        initial_positions = initial_joint_positions,
        target_positions = joint_positions,
        collision_bodies = sim.get_collision_ids(),
        seed = 0,
        physics_client_id = sim.physics_client_id,
        held_object=sim.held_object_id,
        base_link_to_held_obj=sim.held_object_tf,
    )

    if direct_path:
        # Rajat ToDo: Discuss arm / robot dissociation with Tom
        return _plan_to_sim_state_trajectory(direct_path, sim), [JointCommand(pos=joint_positions[:7])]
                                                                 
    print("No direct path found. Running motion planning.")
    plan = run_motion_planning(robot = sim.robot,
        initial_positions = initial_joint_positions,
        target_positions = joint_positions,
        collision_bodies = sim.get_collision_ids(),
        seed = 0,
        physics_client_id = sim.physics_client_id,
        held_object=sim.held_object_id,
        base_link_to_held_obj=sim.held_object_tf,
    )
    plan = _plan_to_sim_state_trajectory(plan, sim)
    remapped_plan = remap_trajectory_to_constant_distance(plan, sim)

    return remapped_plan, simulated_trajectory_to_kinova_commands(remapped_plan)

def teleport_to_ee_pose(
    sim: FeedingDeploymentPyBulletSimulator,
    pose: Pose,
    expected_joint_positions: list[float],
) -> tuple[list[FeedingDeploymentSimulatorState], list[KinovaCommand]]:
    """
    Call Kinova's move_to_ee_pose to move the robot to the specified pose.
    We do not yet know the implementation of move_to_ee_pose, so we we teleport in simulation.
    """

    command = CartesianCommand(pos=pose.position, quat=pose.orientation)

    cup_pose = sim.scene_description.cup_pose
    wiper_pose = sim.scene_description.wiper_pose
    utensil_pose = sim.scene_description.utensil_inside_mount

    if sim.held_object_name is not None:
        if sim.held_object_name == 'cup':
            cup_pose = None
        elif sim.held_object_name == 'wiper':
            wiper_pose = None
        elif sim.held_object_name == 'utensil':
            utensil_pose = None

    sim_state = FeedingDeploymentSimulatorState(robot_joints=expected_joint_positions,
        cup_pose=cup_pose,
        wiper_pose=wiper_pose,
        utensil_pose=utensil_pose,
        held_object=sim.held_object_name,
        held_object_tf=sim.held_object_tf,
    )
    sim.sync(sim_state)

    return [sim_state], [command]
    