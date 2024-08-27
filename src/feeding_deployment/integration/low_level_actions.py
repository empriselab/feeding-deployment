"""Low-level actions that we can simulate and execute."""

import abc
from copy import copy, deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose
from pybullet_helpers.motion_planning import (
    get_joint_positions_distance,
    run_motion_planning,
    run_smooth_motion_planning_to_pose,
    smoothly_follow_end_effector_path,
    try_direct_path,
)

from feeding_deployment.integration.perception_interface import PerceptionInterface
from feeding_deployment.integration.utils import simulated_trajectory_to_kinova_commands
from feeding_deployment.robot_controller.arm_client import Arm
from feeding_deployment.robot_controller.command_interface import (
    CartesianCommand,
    CloseGripperCommand,
    JointCommand,
    KinovaCommand,
    OpenGripperCommand,
)
from feeding_deployment.simulation.planning import (
    _get_motion_plan_for_robot_finger_tip,
    _get_plan_to_execute_grasp,
    _plan_to_sim_state_trajectory,
    get_bite_transfer_plan,
    get_plan_to_grasp_cup,
    get_plan_to_grasp_utensil,
    get_plan_to_grasp_wiper,
    get_plan_to_stow_cup,
    get_plan_to_stow_utensil,
    get_plan_to_stow_wiper,
    get_plan_to_transfer_cup,
    get_plan_to_transfer_wiper,
    remap_trajectory_to_constant_distance,
)
from feeding_deployment.simulation.simulator import FeedingDeploymentPyBulletSimulator
from feeding_deployment.simulation.state import FeedingDeploymentSimulatorState


def move_to_joint_positions(
    sim: FeedingDeploymentPyBulletSimulator,
    joint_positions: list[float],
    sim_states: list[FeedingDeploymentSimulatorState],
    robot_commands: list[KinovaCommand],
) -> None:
    """Move the robot to the specified joint positions."""

    initial_joint_positions = sim.robot.get_joint_positions().copy()
    target_joint_positions = joint_positions.copy()
    
    # Add current gripper joint positions to the target joint positions
    target_joint_positions.append(initial_joint_positions[-1])

    # Rajat ToDo: Check if I need to collsion check initial and target positions before calling this function
    direct_path = try_direct_path(
        robot=sim.robot,
        initial_positions=initial_joint_positions,
        target_positions=target_joint_positions,
        collision_bodies=sim.get_collision_ids(),
        physics_client_id=sim.physics_client_id,
        held_object=sim.held_object_id,
        base_link_to_held_obj=sim.held_object_tf,
    )

    if direct_path:
        # Rajat ToDo: Discuss arm / robot dissociation with Tom
        sim_states.extend(_plan_to_sim_state_trajectory(direct_path, sim))
        robot_commands.append(JointCommand(pos=target_joint_positions[:7]))
        return

    print("No direct path found. Running motion planning.")
    plan = run_motion_planning(
        robot=sim.robot,
        initial_positions=initial_joint_positions,
        target_positions=target_joint_positions,
        collision_bodies=sim.get_collision_ids(),
        seed=0,
        physics_client_id=sim.physics_client_id,
        held_object=sim.held_object_id,
        base_link_to_held_obj=sim.held_object_tf,
    )
    plan = _plan_to_sim_state_trajectory(plan, sim)
    remapped_plan = remap_trajectory_to_constant_distance(plan, sim)

    sim_states.extend(remapped_plan)
    robot_commands.extend(simulated_trajectory_to_kinova_commands(remapped_plan))


def teleport_to_ee_pose(
    sim: FeedingDeploymentPyBulletSimulator,
    pose: Pose,
    expected_joint_positions: list[float],
    sim_states: list[FeedingDeploymentSimulatorState],
    robot_commands: list[KinovaCommand],
) -> None:
    """Call Kinova's move_to_ee_pose to move the robot to the specified pose.

    We do not yet know the implementation of move_to_ee_pose, so we we
    teleport in simulation.
    """
    command = CartesianCommand(pos=pose.position, quat=pose.orientation)

    cup_pose = sim.scene_description.cup_pose
    wiper_pose = sim.scene_description.wiper_pose
    utensil_pose = sim.scene_description.utensil_inside_mount

    if sim.held_object_name is not None:
        if sim.held_object_name == "cup":
            cup_pose = None
        elif sim.held_object_name == "wiper":
            wiper_pose = None
        elif sim.held_object_name == "utensil":
            utensil_pose = None

    sim_state = FeedingDeploymentSimulatorState(
        robot_joints=expected_joint_positions + [sim.robot.get_joint_positions()[-1]], # add gripper joint
        cup_pose=cup_pose,
        wiper_pose=wiper_pose,
        utensil_pose=utensil_pose,
        held_object=sim.held_object_name,
        held_object_tf=sim.held_object_tf,
    )
    sim.sync(sim_state)

    sim_states.append(sim_state)
    robot_commands.append(command)

def move_to_ee_pose(
    sim: FeedingDeploymentPyBulletSimulator,
    target_pose: Pose,
    exclude_collision_ids: set[int] | None,
    tip_from_end_effector: Pose,
    max_motion_plan_time: float,
    sim_states: list[FeedingDeploymentSimulatorState],
    robot_commands: list[KinovaCommand],
) -> None:
    """Plan ee pose trajectory to desired pose."""

    # Commands will be in end effector space, but grasp planning will be in
    # tip space.
    robot = sim.robot
    physics_client_id = sim.physics_client_id

    # Run motion planning.
    collision_ids = sim.get_collision_ids()
    if exclude_collision_ids is not None:
        collision_ids -= exclude_collision_ids

    plan = run_smooth_motion_planning_to_pose(
        target_pose,
        robot,
        collision_ids,
        tip_from_end_effector,
        seed=0,
        max_time=max_motion_plan_time,
        held_object=sim.held_object_id,
        base_link_to_held_obj=sim.held_object_tf,
    )
    assert plan is not None

    plan = _plan_to_sim_state_trajectory(plan, sim)
    remapped_plan = remap_trajectory_to_constant_distance(plan, sim)

    sim_states.extend(remapped_plan)
    robot_commands.extend(simulated_trajectory_to_kinova_commands(remapped_plan))

