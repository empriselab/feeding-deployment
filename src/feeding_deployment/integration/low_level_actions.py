"""Low-level actions that we can simulate and execute."""

import abc
from copy import copy, deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose
from pybullet_helpers.inverse_kinematics import add_fingers_to_joint_positions
from pybullet_helpers.motion_planning import (
    get_joint_positions_distance,
    run_motion_planning,
    run_smooth_motion_planning_to_pose,
    smoothly_follow_end_effector_path,
)
from pybullet_helpers.gui import visualize_pose

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
    _plan_to_sim_state_trajectory,
    remap_trajectory_to_constant_distance,
    run_smooth_ee_interpolated_planning_to_pose
)
from feeding_deployment.simulation.simulator import FeedingDeploymentPyBulletSimulator
from feeding_deployment.simulation.state import FeedingDeploymentSimulatorState


def move_to_joint_positions(
    sim: FeedingDeploymentPyBulletSimulator,
    joint_positions: list[float],
    sim_states: list[FeedingDeploymentSimulatorState],
    robot_commands: list[KinovaCommand],
) -> None:
    """Move the robot to the specified joint positions.

    NOTE: joint_positions does NOT include finger joints.
    """

    initial_joint_positions = sim.robot.get_joint_positions().copy()
    target_joint_positions = add_fingers_to_joint_positions(sim.robot, joint_positions)

    direct_path = run_motion_planning(
        robot=sim.robot,
        initial_positions=initial_joint_positions,
        target_positions=target_joint_positions,
        collision_bodies=sim.get_collision_ids(),
        seed=0,  # not used
        physics_client_id=sim.physics_client_id,
        held_object=sim.held_object_id,
        base_link_to_held_obj=sim.held_object_tf,
        direct_path_only=True,
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

    NOTE: expected_joint_positions does NOT include finger joints.
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

    target_joint_positions = add_fingers_to_joint_positions(
        sim.robot, expected_joint_positions
    )
    sim_state = FeedingDeploymentSimulatorState(
        robot_joints=target_joint_positions,
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
    check_held_object_collisions: bool = True,
) -> None:
    """Plan ee pose trajectory to desired pose."""

    # input("In move_to_ee_pose")

    # Run motion planning.
    plan = run_smooth_ee_interpolated_planning_to_pose(
        target_pose,
        tip_from_end_effector.invert(),
        sim,
        num_interp_waypoints=10,
        max_motion_plan_time=max_motion_plan_time,
        exclude_collision_ids=exclude_collision_ids,
        check_held_object_collisions=check_held_object_collisions,
    )

    assert plan is not None

    remapped_plan = remap_trajectory_to_constant_distance(plan, sim)

    sim_states.extend(remapped_plan)
    robot_commands.extend(simulated_trajectory_to_kinova_commands(remapped_plan))

