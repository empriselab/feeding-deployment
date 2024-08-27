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

###############################################################################
#                           Assisted Drinking                                 #
###############################################################################


def get_plan_to_grasp_cup(
    sim: FeedingDeploymentPyBulletSimulator,
    seed: int = 0,
    max_motion_plan_time: float = 10.0,
    num_grasp_waypoints: int = 5,
    num_prestow_waypoints: int = 3,
    num_staging_interp: int = 25,
) -> list[FeedingDeploymentSimulatorState]:
    """Make a plan to grasp the cup from the current simulator state."""

    assert sim.held_object_name is None

    # Quiet IKfast warnings.
    logging.disable(logging.ERROR)

    sim_states: list[FeedingDeploymentSimulatorState] = []

    # Move to pregrasp.
    sim_states.extend(
        _get_motion_plan_for_robot_finger_tip(
            sim.scene_description.cup_pregrasp_pose,
            sim,
            seed,
            max_motion_plan_time,
        )
    )

    # Move to grasp.
    sim_states.extend(
        _get_interpolated_plan_for_robot_finger_tip(
            sim.scene_description.cup_grasp_pose,
            sim,
            num_grasp_waypoints,
            max_motion_plan_time,
            exclude_collision_ids={sim.cup_id},
        )
    )

    # Execute the grasp.
    sim_states.extend(_get_plan_to_execute_grasp(sim, "cup"))

    # Move to prestow.
    sim_states.extend(
        _get_interpolated_plan_for_robot_finger_tip(
            sim.scene_description.cup_prestow_pose,
            sim,
            num_prestow_waypoints,
            max_motion_plan_time,
            exclude_collision_ids={sim.table_id},
        )
    )
    # Move to staging.
    sim_states.extend(
        _get_interpolated_plan_for_robot_finger_tip(
            sim.scene_description.cup_staging_pose,
            sim,
            num_staging_interp,
            max_motion_plan_time,
        )
    )

    return sim_states


def get_plan_to_stow_cup(
    sim: FeedingDeploymentPyBulletSimulator,
    max_motion_plan_time: float = 10.0,
    num_grasp_waypoints: int = 5,
    num_prestow_waypoints: int = 25,
) -> list[FeedingDeploymentSimulatorState]:
    """Make a plan to stow the cup from the current simulator state."""

    # Quiet IKfast warnings.
    logging.disable(logging.ERROR)

    sim_states: list[FeedingDeploymentSimulatorState] = []

    # Move to prestow.
    sim_states.extend(
        _get_interpolated_plan_for_robot_finger_tip(
            sim.scene_description.cup_prestow_pose,
            sim,
            num_prestow_waypoints,
            max_motion_plan_time,
        )
    )

    # Move to the release point.
    sim_states.extend(
        _get_interpolated_plan_for_robot_finger_tip(
            sim.scene_description.cup_grasp_pose,
            sim,
            num_grasp_waypoints,
            max_motion_plan_time,
            exclude_collision_ids={sim.table_id},
        )
    )

    # Close to grasp.
    sim_states.extend(_get_plan_to_execute_ungrasp(sim))

    # Move to pregrasp.
    sim_states.extend(
        _get_interpolated_plan_for_robot_finger_tip(
            sim.scene_description.cup_pregrasp_pose,
            sim,
            num_grasp_waypoints,
            max_motion_plan_time,
            exclude_collision_ids={sim.cup_id},
        )
    )

    return sim_states


def get_plan_to_transfer_cup(
    sim: FeedingDeploymentPyBulletSimulator,
    max_motion_plan_time: float = 10.0,
    num_grasp_waypoints: int = 5,
) -> list[FeedingDeploymentSimulatorState]:
    """Make a plan to transfer the cup, assuming it's near staging."""

    assert sim.held_object_name == "cup"

    # Quiet IKfast warnings.
    logging.disable(logging.ERROR)

    sim_states: list[FeedingDeploymentSimulatorState] = []

    # Move to transfer.
    sim_states.extend(
        _get_interpolated_plan_for_robot_finger_tip(
            sim.scene_description.cup_transfer_pose,
            sim,
            num_grasp_waypoints,
            max_motion_plan_time,
        )
    )

    return sim_states


###############################################################################
#                             Assisted Wiping                                 #
###############################################################################


def get_plan_to_grasp_wiper(
    sim: FeedingDeploymentPyBulletSimulator,
    seed: int = 0,
    max_motion_plan_time: float = 10.0,
    num_grasp_waypoints: int = 5,
    num_prestow_waypoints: int = 3,
    num_staging_interp: int = 25,
) -> list[FeedingDeploymentSimulatorState]:
    """Make a plan to grasp the wiper from the current simulator state."""

    assert sim.held_object_name is None

    # Quiet IKfast warnings.
    logging.disable(logging.ERROR)

    sim_states: list[FeedingDeploymentSimulatorState] = []

    # Move to pregrasp.
    sim_states.extend(
        _get_motion_plan_for_robot_finger_tip(
            sim.scene_description.wiper_pregrasp_pose,
            sim,
            seed,
            max_motion_plan_time,
        )
    )

    # Move to grasp.
    sim_states.extend(
        _get_interpolated_plan_for_robot_finger_tip(
            sim.scene_description.wiper_grasp_pose,
            sim,
            num_grasp_waypoints,
            max_motion_plan_time,
            exclude_collision_ids={sim.wiper_id},
        )
    )

    # Execute the grasp.
    sim_states.extend(_get_plan_to_execute_grasp(sim, "wiper"))

    # Move to prestow.
    sim_states.extend(
        _get_interpolated_plan_for_robot_finger_tip(
            sim.scene_description.wiper_prestow_pose,
            sim,
            num_prestow_waypoints,
            max_motion_plan_time,
        )
    )

    # Move to staging.
    sim_states.extend(
        _get_interpolated_plan_for_robot_finger_tip(
            sim.scene_description.wiper_staging_pose,
            sim,
            num_staging_interp,
            max_motion_plan_time,
        )
    )

    return sim_states


def get_plan_to_stow_wiper(
    sim: FeedingDeploymentPyBulletSimulator,
    max_motion_plan_time: float = 10.0,
    num_grasp_waypoints: int = 5,
    num_prestow_waypoints: int = 25,
) -> list[FeedingDeploymentSimulatorState]:
    """Make a plan to stow the wiper from the current simulator state."""

    # Quiet IKfast warnings.
    logging.disable(logging.ERROR)

    sim_states: list[FeedingDeploymentSimulatorState] = []

    # Move to prestow.
    sim_states.extend(
        _get_interpolated_plan_for_robot_finger_tip(
            sim.scene_description.wiper_prestow_pose,
            sim,
            num_prestow_waypoints,
            max_motion_plan_time,
        )
    )

    # Move to the release point.
    sim_states.extend(
        _get_interpolated_plan_for_robot_finger_tip(
            sim.scene_description.wiper_grasp_pose,
            sim,
            num_grasp_waypoints,
            max_motion_plan_time,
        )
    )

    # Close to grasp.
    sim_states.extend(_get_plan_to_execute_ungrasp(sim))

    # Move to pregrasp.
    sim_states.extend(
        _get_interpolated_plan_for_robot_finger_tip(
            sim.scene_description.wiper_pregrasp_pose,
            sim,
            num_grasp_waypoints,
            max_motion_plan_time,
            exclude_collision_ids={sim.wiper_id},
        )
    )

    return sim_states


def get_plan_to_transfer_wiper(
    sim: FeedingDeploymentPyBulletSimulator,
    max_motion_plan_time: float = 10.0,
    num_grasp_waypoints: int = 5,
) -> list[FeedingDeploymentSimulatorState]:
    """Make a plan to transfer the wiper, assuming it's near staging."""

    assert sim.held_object_name == "wiper"

    # Quiet IKfast warnings.
    logging.disable(logging.ERROR)

    sim_states: list[FeedingDeploymentSimulatorState] = []

    # Move to transfer.
    sim_states.extend(
        _get_interpolated_plan_for_robot_finger_tip(
            sim.scene_description.wiper_transfer_pose,
            sim,
            num_grasp_waypoints,
            max_motion_plan_time,
        )
    )

    return sim_states


###############################################################################
#                            Assisted Feeding                                 #
###############################################################################


def get_plan_to_grasp_utensil(
    sim: FeedingDeploymentPyBulletSimulator,
    seed: int = 0,
    max_motion_plan_time: float = 10.0,
    num_grasp_waypoints: int = 5,
    num_prestow_waypoints: int = 3,
    num_staging_interp: int = 25,
) -> list[FeedingDeploymentSimulatorState]:
    """Make a plan to grasp the utensil from the current simulator state."""

    assert sim.held_object_name is None

    # Quiet IKfast warnings.
    logging.disable(logging.ERROR)

    sim_states: list[FeedingDeploymentSimulatorState] = []

    # TODO remove
    # import pybullet as p
    # from pybullet_helpers.gui import visualize_pose
    # visualize_pose(sim.scene_description.utensil_pregrasp_pose, sim.physics_client_id)
    # while True:
    #     p.stepSimulation(sim.physics_client_id)

    # Move to pregrasp.
    sim_states.extend(
        _get_motion_plan_for_robot_finger_tip(
            sim.scene_description.utensil_pregrasp_pose,
            sim,
            seed,
            max_motion_plan_time,
        )
    )

    # Move to grasp.
    sim_states.extend(
        _get_interpolated_plan_for_robot_finger_tip(
            sim.scene_description.utensil_grasp_pose,
            sim,
            num_grasp_waypoints,
            max_motion_plan_time,
            exclude_collision_ids={sim.utensil_id},
        )
    )

    # Execute the grasp.
    sim_states.extend(_get_plan_to_execute_grasp(sim, "utensil"))

    # Move to prestow.
    sim_states.extend(
        _get_interpolated_plan_for_robot_finger_tip(
            sim.scene_description.utensil_prestow_pose,
            sim,
            num_prestow_waypoints,
            max_motion_plan_time,
        )
    )

    # Move to a special intermediate waypoint to avoid self-collisions that
    # tend to arise with the utensil (because it is in the back).
    sim_states.extend(
        _get_interpolated_plan_for_robot_finger_tip(
            sim.scene_description.utensil_corner_waypoint_pose,
            sim,
            num_prestow_waypoints,
            max_motion_plan_time,
        )
    )

    # Move to staging.
    sim_states.extend(
        _get_interpolated_plan_for_robot_finger_tip(
            sim.scene_description.utensil_staging_pose,
            sim,
            num_staging_interp,
            max_motion_plan_time,
        )
    )

    return sim_states


def get_plan_to_stow_utensil(
    sim: FeedingDeploymentPyBulletSimulator,
    max_motion_plan_time: float = 10.0,
    num_grasp_waypoints: int = 5,
    num_prestow_waypoints: int = 25,
) -> list[FeedingDeploymentSimulatorState]:
    """Make a plan to stow the utensil from the current simulator state."""

    # Quiet IKfast warnings.
    logging.disable(logging.ERROR)

    sim_states: list[FeedingDeploymentSimulatorState] = []

    # Move to a special intermediate waypoint to avoid self-collisions that
    # tend to arise with the utensil (because it is in the back).
    sim_states.extend(
        _get_interpolated_plan_for_robot_finger_tip(
            sim.scene_description.utensil_corner_waypoint_pose,
            sim,
            num_prestow_waypoints,
            max_motion_plan_time,
        )
    )

    # Move to prestow.
    sim_states.extend(
        _get_interpolated_plan_for_robot_finger_tip(
            sim.scene_description.utensil_prestow_pose,
            sim,
            num_prestow_waypoints,
            max_motion_plan_time,
        )
    )

    # Move to the release point.
    sim_states.extend(
        _get_interpolated_plan_for_robot_finger_tip(
            sim.scene_description.utensil_grasp_pose,
            sim,
            num_grasp_waypoints,
            max_motion_plan_time,
        )
    )

    # Close to grasp.
    sim_states.extend(_get_plan_to_execute_ungrasp(sim))

    # Move to pregrasp.
    sim_states.extend(
        _get_interpolated_plan_for_robot_finger_tip(
            sim.scene_description.utensil_pregrasp_pose,
            sim,
            num_grasp_waypoints,
            max_motion_plan_time,
            exclude_collision_ids={sim.utensil_id},
        )
    )

    return sim_states


###############################################################################
#                             Bite Transfer                                   #
###############################################################################


def get_bite_transfer_plan(
    forque_target_pose: Pose,
    sim: FeedingDeploymentPyBulletSimulator,
    max_motion_plan_time: float = 10.0,
) -> list[FeedingDeploymentSimulatorState]:
    """Plan to transfer a bite, assuming we're already prepared to do so."""

    robot = sim.robot
    physics_client_id = sim.physics_client_id
    collision_ids = sim.get_collision_ids()

    # Create joint distance function.
    _joint_distance_fn = _create_joint_distance_fn(sim.robot)

    finger_frame_id = robot.link_from_name("finger_tip")
    end_effector_link_id = robot.link_from_name(robot.tool_link_name)
    finger_from_end_effector = get_relative_link_pose(
        robot.robot_id, end_effector_link_id, finger_frame_id, physics_client_id
    )

    # Rajat ToDo: switch to forque_target_pose orientation
    current_end_effector_pose = robot.get_end_effector_pose()
    current_fingers_pose = get_link_pose(
        robot.robot_id, finger_frame_id, physics_client_id
    )
    bite_transfer_fingers_pose: Pose = Pose(
        forque_target_pose.position, current_fingers_pose.orientation
    )

    new_end_effector_pose = multiply_poses(
        bite_transfer_fingers_pose, finger_from_end_effector
    )
    interpolated_poses = list(
        interpolate_poses(
            current_end_effector_pose,
            new_end_effector_pose,
            num_interp=10,  # NOTE
        )
    )

    plan = smoothly_follow_end_effector_path(
        robot,
        interpolated_poses,
        robot.get_joint_positions(),
        collision_ids,
        _joint_distance_fn,
        max_time=max_motion_plan_time,
    )

    # TODO update once we model the utensil in sim
    return [FeedingDeploymentSimulatorState(joints) for joints in plan]


###############################################################################
#                           General Functions                                 #
###############################################################################


def _get_motion_plan_for_robot_finger_tip(
    target_pose: Pose,
    sim: FeedingDeploymentPyBulletSimulator,
    seed: int,
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

    # Run motion planning.
    collision_ids = sim.get_collision_ids()
    if exclude_collision_ids is not None:
        collision_ids -= exclude_collision_ids

    plan = run_smooth_motion_planning_to_pose(
        target_pose,
        robot,
        collision_ids,
        finger_from_end_effector.invert(),
        seed,
        max_time=max_motion_plan_time,
        held_object=sim.held_object_id,
        base_link_to_held_obj=sim.held_object_tf,
    )
    assert plan is not None
    return _plan_to_sim_state_trajectory(plan, sim)


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


def _plan_to_sim_state_trajectory(
    plan: list[JointPositions], sim: FeedingDeploymentPyBulletSimulator
) -> list[FeedingDeploymentSimulatorState]:
    # Read out the simulator states from the plan.
    cup_pose: Pose | None = None
    if sim.held_object_name != "cup":
        cup_pose = get_pose(sim.cup_id, sim.physics_client_id)
    wiper_pose: Pose | None = None
    if sim.held_object_name != "wiper":
        wiper_pose = get_pose(sim.wiper_id, sim.physics_client_id)
    utensil_pose: Pose | None = None
    if sim.held_object_name != "utensil":
        utensil_pose = get_pose(sim.utensil_id, sim.physics_client_id)

    sim_states: list[FeedingDeploymentSimulatorState] = []
    for joints in plan:
        sim_state = FeedingDeploymentSimulatorState(
            joints,
            cup_pose=cup_pose,
            wiper_pose=wiper_pose,
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
    robot.open_fingers()
    sim.held_object_name = object_name
    if object_name == "cup":
        sim.held_object_id = sim.cup_id
    elif object_name == "wiper":
        sim.held_object_id = sim.wiper_id
    elif object_name == "utensil":
        sim.held_object_id = sim.utensil_id
    else:
        raise NotImplementedError("TODO")
    world_from_end_effector = get_link_pose(
        robot.robot_id, robot.end_effector_id, physics_client_id
    )
    assert sim.held_object_id is not None
    world_from_held_object = get_pose(sim.held_object_id, physics_client_id)
    base_link_to_held_obj = multiply_poses(
        world_from_end_effector.invert(), world_from_held_object
    )
    sim.held_object_tf = base_link_to_held_obj
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
            cup_pose=s0.cup_pose,
            wiper_pose=s0.wiper_pose,
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
