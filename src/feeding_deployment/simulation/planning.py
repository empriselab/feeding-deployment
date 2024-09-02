"""Functions for planning using the feeding deployment simulator."""

from __future__ import annotations

import logging
from typing import Callable

from pybullet_helpers.geometry import Pose, get_pose, interpolate_poses, multiply_poses
from pybullet_helpers.inverse_kinematics import (
    end_effector_transform_to_joints,
    set_robot_joints_with_held_object,
)
from pybullet_helpers.joint import JointPositions, get_joint_infos, interpolate_joints
from pybullet_helpers.link import get_link_pose, get_relative_link_pose
from pybullet_helpers.math_utils import geometric_sequence
from pybullet_helpers.motion_planning import (
    get_joint_positions_distance,
    run_smooth_motion_planning_to_pose,
    smoothly_follow_end_effector_path,
)
from pybullet_helpers.robots import SingleArmPyBulletRobot
from pybullet_helpers.trajectory import (
    TrajectorySegment,
    concatenate_trajectories,
    iter_traj_with_max_distance,
)

from feeding_deployment.simulation.simulator import FeedingDeploymentPyBulletSimulator
from feeding_deployment.simulation.state import FeedingDeploymentSimulatorState


def _get_interpolated_plan_for_robot_finger_tip(
    target_pose: Pose,
    sim: FeedingDeploymentPyBulletSimulator,
    num_interp_waypoints: int,
    max_motion_plan_time: float,
    exclude_collision_ids: set[int] | None = None,
) -> list[FeedingDeploymentSimulatorState]:

    # Commands will be in end effector space, but grasp planning will be in
    # finger frame space.
    robot = sim.robot
    physics_client_id = sim.physics_client_id
    finger_frame_id = robot.link_from_name("finger_tip")
    end_effector_link_id = robot.link_from_name(robot.tool_link_name)
    finger_from_end_effector = get_relative_link_pose(
        robot.robot_id, end_effector_link_id, finger_frame_id, physics_client_id
    )

    target_end_effector_pose = multiply_poses(target_pose, finger_from_end_effector)
    current_end_effector_pose = robot.get_end_effector_pose()

    interpolated_poses = list(
        interpolate_poses(
            current_end_effector_pose,
            target_end_effector_pose,
            num_interp=num_interp_waypoints,
        )
    )

    _joint_distance_fn = _create_joint_distance_fn(robot)

    collision_ids = sim.get_collision_ids()
    if exclude_collision_ids is not None:
        collision_ids -= exclude_collision_ids

    plan = smoothly_follow_end_effector_path(
        robot,
        interpolated_poses,
        robot.get_joint_positions(),
        collision_ids,
        _joint_distance_fn,
        max_time=max_motion_plan_time,
        held_object=sim.held_object_id,
        base_link_to_held_obj=sim.held_object_tf,
    )
    return _plan_to_sim_state_trajectory(plan, sim)


def run_smooth_ee_interpolated_planning_to_pose(
    target_pose: Pose,
    end_effector_from_tip: Pose,
    sim: FeedingDeploymentPyBulletSimulator,
    num_interp_waypoints: int,
    max_motion_plan_time: float,
    exclude_collision_ids: set[int] | None = None,
    check_held_object_collisions: bool = True,
) -> list[FeedingDeploymentSimulatorState]:

    # Commands will be in end effector space, but grasp planning will be in
    # finger frame space.
    robot = sim.robot

    target_end_effector_pose = multiply_poses(target_pose, end_effector_from_tip)
    current_end_effector_pose = robot.get_end_effector_pose()

    interpolated_poses = list(
        interpolate_poses(
            current_end_effector_pose,
            target_end_effector_pose,
            num_interp=num_interp_waypoints,
        )
    )

    _joint_distance_fn = _create_joint_distance_fn(robot)

    collision_ids = sim.get_collision_ids()
    if exclude_collision_ids is not None:
        collision_ids -= exclude_collision_ids

    if check_held_object_collisions:
        held_object = sim.held_object_id
        base_link_to_held_obj = sim.held_object_tf
    else:
        held_object = None
        base_link_to_held_obj = None

    plan = smoothly_follow_end_effector_path(
        robot,
        interpolated_poses,
        robot.get_joint_positions(),
        collision_ids,
        _joint_distance_fn,
        max_time=max_motion_plan_time,
        held_object=held_object,
        base_link_to_held_obj=base_link_to_held_obj,
    )
    return _plan_to_sim_state_trajectory(plan, sim)

def _plan_to_sim_state_trajectory(
    plan: list[JointPositions], sim: FeedingDeploymentPyBulletSimulator
) -> list[FeedingDeploymentSimulatorState]:
    # Read out the simulator states from the plan.
    drink_pose: Pose | None = None
    if sim.held_object_name != "drink":
        drink_pose = get_pose(sim.drink_id, sim.physics_client_id)
    wipe_pose: Pose | None = None
    if sim.held_object_name != "wipe":
        wipe_pose = get_pose(sim.wipe_id, sim.physics_client_id)
    utensil_pose: Pose | None = None
    if sim.held_object_name != "utensil":
        utensil_pose = get_pose(sim.utensil_id, sim.physics_client_id)

    sim_states: list[FeedingDeploymentSimulatorState] = []
    for joints in plan:
        sim_state = FeedingDeploymentSimulatorState(
            joints,
            drink_pose=drink_pose,
            wipe_pose=wipe_pose,
            utensil_pose=utensil_pose,
            held_object=sim.held_object_name,
            held_object_tf=sim.held_object_tf,
        )
        sim_states.append(sim_state)
    # Sync simulator to end of plan.
    sim.sync(sim_states[-1])
    return sim_states


def _get_plan_to_execute_grasp(
    sim: FeedingDeploymentPyBulletSimulator, object_name: str
) -> list[FeedingDeploymentSimulatorState]:

    # Simulate grasping by faking a constraint with the held object.
    robot = sim.robot
    physics_client_id = sim.physics_client_id
    robot.set_finger_state(sim.scene_description.tool_grasp_fingers_value)
    sim.held_object_name = object_name
    if object_name == "drink":
        sim.held_object_id = sim.drink_id
    elif object_name == "wipe":
        sim.held_object_id = sim.wipe_id
    elif object_name == "utensil":
        sim.held_object_id = sim.utensil_id
    else:
        raise NotImplementedError("TODO")

    assert sim.held_object_id is not None
    finger_frame_id = sim.robot.link_from_name("finger_tip")
    end_effector_link_id = sim.robot.link_from_name(sim.robot.tool_link_name)
    finger_from_end_effector = get_relative_link_pose(
        sim.robot.robot_id, finger_frame_id, end_effector_link_id, sim.physics_client_id
    )
    sim.held_object_tf = finger_from_end_effector
    return _plan_to_sim_state_trajectory([robot.get_joint_positions()], sim)


def _get_plan_to_execute_ungrasp(
    sim: FeedingDeploymentPyBulletSimulator,
) -> list[FeedingDeploymentSimulatorState]:
    robot = sim.robot
    robot.close_fingers()
    sim.held_object_name = None
    sim.held_object_tf = None
    sim.held_object_id = None
    return _plan_to_sim_state_trajectory([robot.get_joint_positions()], sim)


def _create_joint_distance_fn(
    robot: SingleArmPyBulletRobot,
) -> Callable[[JointPositions, JointPositions], float]:

    weights = geometric_sequence(0.9, len(robot.arm_joint_names))
    joint_infos = get_joint_infos(
        robot.robot_id, robot.arm_joints, robot.physics_client_id
    )

    def _joint_distance_fn(pt1: JointPositions, pt2: JointPositions) -> float:
        return get_joint_positions_distance(
            robot,
            joint_infos,
            pt1,
            pt2,
            metric="weighted_joints",
            weights=weights,
        )

    return _joint_distance_fn


def remap_trajectory_to_constant_distance(
    traj: list[FeedingDeploymentSimulatorState],
    sim: FeedingDeploymentPyBulletSimulator,
    max_joint_space_distance: float = 0.1,
) -> list[FeedingDeploymentSimulatorState]:
    """Remap a trajectory so that joint waypoints have constant distance."""

    robot = sim.robot
    joint_infos = get_joint_infos(
        robot.robot_id, robot.arm_joints, robot.physics_client_id
    )
    _joint_distance_fn = _create_joint_distance_fn(robot)

    # Create a continuous-time trajectory.
    def _interpolate_fn(
        s0: FeedingDeploymentSimulatorState,
        s1: FeedingDeploymentSimulatorState,
        t: float,
    ) -> FeedingDeploymentSimulatorState:
        # Interpolate the robot joints.
        robot_joints = interpolate_joints(
            joint_infos, s0.robot_joints, s1.robot_joints, t
        )
        # Interpolate the movable object poses.
        # TODO need to refactor interpolate_poses.
        return FeedingDeploymentSimulatorState(
            robot_joints,
            drink_pose=s0.drink_pose,
            wipe_pose=s0.wipe_pose,
            utensil_pose=s0.utensil_pose,
            held_object=s0.held_object,
            held_object_tf=s0.held_object_tf,
        )

    def _distance_fn(
        s0: FeedingDeploymentSimulatorState, s1: FeedingDeploymentSimulatorState
    ) -> float:
        return _joint_distance_fn(s0.robot_joints, s1.robot_joints)

    # Use distances as times.
    distances = []
    for pt1, pt2 in zip(traj[:-1], traj[1:], strict=True):
        dist = _distance_fn(pt1, pt2)
        distances.append(dist)

    segments = []
    for t in range(len(traj) - 1):
        seg = TrajectorySegment(
            traj[t],
            traj[t + 1],
            distances[t],
            interpolate_fn=_interpolate_fn,
            distance_fn=_distance_fn,
        )
        segments.append(seg)
    continuous_time_trajectory = concatenate_trajectories(segments)
    remapped_traj = list(
        iter_traj_with_max_distance(
            continuous_time_trajectory, max_joint_space_distance
        )
    )
    return remapped_traj
