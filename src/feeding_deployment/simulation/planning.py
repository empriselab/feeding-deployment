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

from feeding_deployment.simulation.world import FeedingDeploymentPyBulletWorld
from feeding_deployment.simulation.state import FeedingDeploymentWorldState

def _plan_to_sim_state_trajectory(
    plan: list[JointPositions], sim: FeedingDeploymentPyBulletWorld
) -> list[FeedingDeploymentWorldState]:
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
    plate_pose: Pose | None = None
    if sim.held_object_name != "plate":
        plate_pose = get_pose(sim.plate_id, sim.physics_client_id)

    sim_states: list[FeedingDeploymentWorldState] = []
    for joints in plan:
        sim_state = FeedingDeploymentWorldState(
            joints,
            drink_pose=drink_pose,
            wipe_pose=wipe_pose,
            utensil_pose=utensil_pose,
            plate_pose=plate_pose,
            held_object=sim.held_object_name,
            held_object_tf=sim.held_object_tf,
        )
        sim_states.append(sim_state)
    # Sync simulator to end of plan.
    if len(sim_states) > 0:
        sim.sync(sim_states[-1])
    return sim_states


def _get_plan_to_execute_grasp(
    sim: FeedingDeploymentPyBulletWorld, object_name: str
) -> list[FeedingDeploymentWorldState]:

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
    elif object_name == "plate":
        sim.held_object_id = sim.plate_id
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
    sim: FeedingDeploymentPyBulletWorld,
) -> list[FeedingDeploymentWorldState]:
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
    traj: list[FeedingDeploymentWorldState],
    sim: FeedingDeploymentPyBulletWorld,
    max_joint_space_distance: float = 0.1,
) -> list[FeedingDeploymentWorldState]:
    """Remap a trajectory so that joint waypoints have constant distance."""

    robot = sim.robot
    joint_infos = get_joint_infos(
        robot.robot_id, robot.arm_joints, robot.physics_client_id
    )
    _joint_distance_fn = _create_joint_distance_fn(robot)

    # Create a continuous-time trajectory.
    def _interpolate_fn(
        s0: FeedingDeploymentWorldState,
        s1: FeedingDeploymentWorldState,
        t: float,
    ) -> FeedingDeploymentWorldState:
        # Interpolate the robot joints.
        robot_joints = interpolate_joints(
            joint_infos, s0.robot_joints, s1.robot_joints, t
        )
        # Interpolate the movable object poses.
        # TODO need to refactor interpolate_poses.
        return FeedingDeploymentWorldState(
            robot_joints,
            drink_pose=s0.drink_pose,
            wipe_pose=s0.wipe_pose,
            utensil_pose=s0.utensil_pose,
            plate_pose=s0.plate_pose,
            held_object=s0.held_object,
            held_object_tf=s0.held_object_tf,
        )

    def _distance_fn(
        s0: FeedingDeploymentWorldState, s1: FeedingDeploymentWorldState
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

    if continuous_time_trajectory.duration < 1e-6:
        logging.warning(
            "Trajectory is too short to remap. "
            "Returning original trajectory."
        )
        return traj

    remapped_traj = list(
        iter_traj_with_max_distance(
            continuous_time_trajectory, max_joint_space_distance
        )
    )
    return remapped_traj
