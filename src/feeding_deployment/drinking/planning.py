"""Script to develop cup manipulation skills in simulation."""

import os
import pickle
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import pybullet as p
from pybullet_helpers.geometry import Pose, get_pose, interpolate_poses, multiply_poses
from pybullet_helpers.gui import create_gui_connection
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
from pybullet_helpers.trajectory import (
    TrajectorySegment,
    concatenate_trajectories,
    iter_traj_with_max_distance,
)

from feeding_deployment.drinking.scene import (
    CupManipulationSceneDescription,
    CupManipulationSceneIDs,
    create_cup_manipulation_scene,
)
from feeding_deployment.drinking.utils import (
    CupManipulationTrajectory,
    make_cup_manipulation_video,
)


def _get_plan_to_pregrasp_cup(
    scene: CupManipulationSceneIDs,
    scene_description: CupManipulationSceneDescription,
    seed: int,
    max_motion_plan_time: float,
) -> CupManipulationTrajectory:

    robot = scene.robot
    physics_client_id = scene.physics_client_id
    assert physics_client_id == robot.physics_client_id
    collision_ids = scene.get_collision_ids()

    # Commands will be in end effector space, but grasp planning will be in
    # finger frame space.
    finger_frame_id = robot.link_from_name("finger_tip")
    end_effector_link_id = robot.link_from_name(robot.tool_link_name)
    finger_from_end_effector = get_relative_link_pose(
        robot.robot_id, end_effector_link_id, finger_frame_id, physics_client_id
    )

    # Find target finger frame pose relative to the cup handle.
    cup_handle_pose = get_link_pose(
        scene.cup_id, scene.cup_handle_link_id, physics_client_id
    )
    cup_grasp = multiply_poses(cup_handle_pose, scene_description.cup_grasp_transform)

    plan = run_smooth_motion_planning_to_pose(
        cup_grasp,
        robot,
        collision_ids,
        finger_from_end_effector,
        seed,
        max_time=max_motion_plan_time,
    )
    assert plan is not None
    return CupManipulationTrajectory(plan, [None] * len(plan))


def _get_plan_to_grasp_cup(
    scene: CupManipulationSceneIDs,
    scene_description: CupManipulationSceneDescription,
    _joint_distance_fn: Callable[[JointPositions, JointPositions], float],
    num_grasp_waypoints: int,
    max_motion_plan_time: float,
) -> CupManipulationTrajectory:

    # Assumes that _get_plan_to_pregrasp_cup() was just called.
    robot = scene.robot
    physics_client_id = scene.physics_client_id
    collision_ids = scene.get_collision_ids(include_cup=False)

    tf = Pose(
        (0.0, 0.0, scene_description.cup_grasp_distance),
        (0.0, 0.0, 0.0, 1.0),
    )
    current_end_effector_pose = robot.get_end_effector_pose()
    new_end_effector_pose = multiply_poses(current_end_effector_pose, tf)
    interpolated_poses = list(
        interpolate_poses(
            current_end_effector_pose,
            new_end_effector_pose,
            num_interp=num_grasp_waypoints,
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
    assert plan is not None
    held_obj_tfs: list[Pose | None] = [None] * len(plan)

    # Open the fingers.
    robot.set_joints(plan[-1])
    robot.open_fingers()
    plan.append(robot.get_joint_positions())
    held_obj_tfs.append(None)

    # Simulate grasping by faking a constraint with the held object.
    robot.set_joints(plan[-1])
    world_from_end_effector = get_link_pose(
        robot.robot_id, robot.end_effector_id, physics_client_id
    )
    world_from_held_object = get_pose(scene.cup_id, physics_client_id)
    base_link_to_held_obj = multiply_poses(
        world_from_end_effector.invert(), world_from_held_object
    )

    # Move off the table so that the cup is no longer in collision with the table.
    tf = Pose((0.0, -0.1, 0.0), (0.0, 0.0, 0.0, 1.0))
    joints = end_effector_transform_to_joints(robot, tf)
    set_robot_joints_with_held_object(
        robot, physics_client_id, scene.cup_id, base_link_to_held_obj, joints
    )

    plan.append(joints)
    held_obj_tfs.append(base_link_to_held_obj)

    return CupManipulationTrajectory(plan, held_obj_tfs)

def _get_move_utensil_to_transfer_plan(
    forque_target_pose: Pose,
    scene: CupManipulationSceneIDs,
    scene_description: CupManipulationSceneDescription,
    _joint_distance_fn: Callable[[JointPositions, JointPositions], float],
    max_motion_plan_time: float,
) -> CupManipulationTrajectory:

    # Assumes that _get_plan_to_grasp_cup() was just called.
    robot = scene.robot
    physics_client_id = scene.physics_client_id
    collision_ids = scene.get_collision_ids(include_cup=False)

    finger_frame_id = robot.link_from_name("finger_tip")
    end_effector_link_id = robot.link_from_name(robot.tool_link_name)
    finger_from_end_effector = get_relative_link_pose(
        robot.robot_id, end_effector_link_id, finger_frame_id, physics_client_id
    )

    transfer_relative_pose: Pose = Pose(
        (0.0, 0.2, 0.0), p.getQuaternionFromEuler((0.0, 0.0, np.pi / 2))
    )
    # Rajat ToDo: switch to forque_target_pose orientation
    current_end_effector_pose = robot.get_end_effector_pose()
    current_fingers_pose = get_link_pose(robot.robot_id, finger_frame_id, physics_client_id)
    # new_fingers_pose = multiply_poses(bite_transfer_pose, fingers_to_bite)
    bite_transfer_fingers_pose: Pose = Pose(
        forque_target_pose.position, current_fingers_pose.orientation
    )

    new_end_effector_pose = multiply_poses(bite_transfer_fingers_pose, finger_from_end_effector)
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

    assert plan is not None
    held_obj_tfs = [None] * len(plan)

    # print type of plan and held_obj_tfs
    print("type of plan: ", type(plan))
    print("type of held_obj_tfs: ", type(held_obj_tfs))
    input("Press enter to continue")

    return CupManipulationTrajectory(plan, held_obj_tfs)

def _get_move_cup_to_staging_plan(
    scene: CupManipulationSceneIDs,
    scene_description: CupManipulationSceneDescription,
    _joint_distance_fn: Callable[[JointPositions, JointPositions], float],
    num_drink_transfer_end_effector_interp: int,
    max_motion_plan_time: float,
    base_link_to_held_obj: Pose,
) -> CupManipulationTrajectory:

    # Assumes that _get_plan_to_grasp_cup() was just called.
    robot = scene.robot
    physics_client_id = scene.physics_client_id
    collision_ids = scene.get_collision_ids(include_cup=False)

    finger_frame_id = robot.link_from_name("finger_tip")
    end_effector_link_id = robot.link_from_name(robot.tool_link_name)
    finger_from_end_effector = get_relative_link_pose(
        robot.robot_id, end_effector_link_id, finger_frame_id, physics_client_id
    )

    new_cup_pose = scene_description.cup_staging_pose
    cup_pose = get_pose(scene.cup_id, physics_client_id)
    current_fingers_pose = get_link_pose(
        robot.robot_id, finger_frame_id, physics_client_id
    )
    fingers_to_cup = multiply_poses(
        cup_pose.invert(),
        current_fingers_pose,
    )
    new_fingers_pose = multiply_poses(new_cup_pose, fingers_to_cup)
    new_end_effector_pose = multiply_poses(new_fingers_pose, finger_from_end_effector)

    # Prevent spilling: interpolate in end effector space and then follow.
    current_end_effector_pose = robot.get_end_effector_pose()
    interpolated_poses = list(
        interpolate_poses(
            current_end_effector_pose,
            new_end_effector_pose,
            num_interp=num_drink_transfer_end_effector_interp,
        )
    )
    plan = smoothly_follow_end_effector_path(
        robot,
        interpolated_poses,
        robot.get_joint_positions(),
        collision_ids,
        _joint_distance_fn,
        max_time=max_motion_plan_time,
        held_object=scene.cup_id,
        base_link_to_held_obj=base_link_to_held_obj,
    )

    return CupManipulationTrajectory(plan, [base_link_to_held_obj] * len(plan))


def _remap_trajectory_to_constant_distance(
    traj: CupManipulationTrajectory,
    scene: CupManipulationSceneIDs,
    _joint_distance_fn: Callable[[JointPositions, JointPositions], float],
    max_joint_space_distance: float,
) -> CupManipulationTrajectory:

    robot = scene.robot
    joint_infos = get_joint_infos(
        robot.robot_id, robot.arm_joints, robot.physics_client_id
    )

    # Create a continuous-time trajectory.
    joint_interpolate_fn = partial(interpolate_joints, joint_infos)
    distances = []
    for pt1, pt2 in zip(traj.joint_states[:-1], traj.joint_states[1:], strict=True):
        dist = _joint_distance_fn(pt1, pt2)
        distances.append(dist)
    # Use distances as times.
    joint_segments = []
    for t in range(len(traj.joint_states) - 1):
        joint_seg = TrajectorySegment(
            traj.joint_states[t],
            traj.joint_states[t + 1],
            distances[t],
            interpolate_fn=joint_interpolate_fn,
            distance_fn=_joint_distance_fn,
        )
        joint_segments.append(joint_seg)
    continuous_time_trajectory = concatenate_trajectories(joint_segments)
    remapped_joint_positions = list(
        iter_traj_with_max_distance(
            continuous_time_trajectory, max_joint_space_distance
        )
    )

    # Remap the cup states.
    def _cup_interpolate_fn(q1: Pose | None, q2: Pose | None, t: float) -> Pose | None:
        del q2, t  # unused
        return q1

    def _cup_distance_fn(q1: Pose | None, q2: Pose | None) -> float:
        raise NotImplementedError

    cup_segments = []
    for t in range(len(traj.held_cup_transforms) - 1):
        cup_seg = TrajectorySegment(
            traj.held_cup_transforms[t],
            traj.held_cup_transforms[t + 1],
            distances[t],
            interpolate_fn=_cup_interpolate_fn,
            distance_fn=_cup_distance_fn,
        )
        cup_segments.append(cup_seg)
    continuous_time_cup_trajectory = concatenate_trajectories(cup_segments)

    ts = np.linspace(
        0,
        continuous_time_trajectory.duration,
        num=len(remapped_joint_positions),
        endpoint=True,
    )
    remapped_held_cup_tfs = [continuous_time_cup_trajectory(t) for t in ts]

    return CupManipulationTrajectory(remapped_joint_positions, remapped_held_cup_tfs)


def generate_trajectory(
    scene: CupManipulationSceneIDs,
    scene_description: CupManipulationSceneDescription,
    max_motion_plan_time: float = 10.0,
    num_grasp_waypoints: int = 5,
    seed: int = 0,
    num_drink_transfer_end_effector_interp: int = 25,
    max_joint_space_distance: float = 0.1,
    force_rerun: bool = False,
) -> CupManipulationTrajectory:
    """Run planning to create a cup manipulation trajectory."""

    saved_traj_dir = Path(__file__).parent / "saved_trajs"
    os.makedirs(saved_traj_dir, exist_ok=True)
    saved_traj_files = list(saved_traj_dir.glob("*.traj"))
    if not saved_traj_files:
        next_file_id = 0
    else:
        next_file_id = len(saved_traj_files)
    next_filepath = saved_traj_dir / f"{next_file_id}.traj"
    assert not next_filepath.exists()

    # Check if we already have saved a trajectory for this scene.
    if not force_rerun:
        for saved_traj_file in saved_traj_files:
            with open(saved_traj_file, "rb") as rfp:
                saved_scene_description, saved_traj = pickle.load(rfp)
            if scene_description.allclose(saved_scene_description):
                traj = saved_traj
                print(f"Loaded saved trajectory: {saved_traj_file.name}")
                return traj

    # Need to replan.
    print("Running planning...")
    plan = CupManipulationTrajectory()

    robot = scene.robot
    physics_client_id = scene.physics_client_id
    assert robot.physics_client_id == physics_client_id

    # Create joint distance function.
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

    pregrasp_cup_plan = _get_plan_to_pregrasp_cup(
        scene, scene_description, seed, max_motion_plan_time
    )
    plan.extend(pregrasp_cup_plan)
    robot.set_joints(plan.joint_states[-1])

    grasp_cup_plan = _get_plan_to_grasp_cup(
        scene,
        scene_description,
        _joint_distance_fn,
        num_grasp_waypoints,
        max_motion_plan_time,
    )
    plan.extend(grasp_cup_plan)
    robot.set_joints(plan.joint_states[-1])

    # Move to staging pose.
    base_link_to_held_obj = grasp_cup_plan.held_cup_transforms[-1]
    assert isinstance(base_link_to_held_obj, Pose)
    move_cup_to_staging_plan = _get_move_cup_to_staging_plan(
        scene,
        scene_description,
        _joint_distance_fn,
        num_drink_transfer_end_effector_interp,
        max_motion_plan_time,
        base_link_to_held_obj,
    )
    plan.extend(move_cup_to_staging_plan)
    robot.set_joints(plan.joint_states[-1])

    # Remap the trajectory.
    plan = _remap_trajectory_to_constant_distance(
        plan, scene, _joint_distance_fn, max_joint_space_distance
    )
    
    # Save the trajectory.
    with open(next_filepath, "wb") as wfp:
        pickle.dump((scene_description, plan), wfp)
    print(f"Dumped trajectory to {next_filepath}")

    return plan

# Rajat ToDo: hacked version of the above function to generate a trajectory for bite transfer
def generate_bite_transfer_trajectory(
    forque_target_pose: Pose,
    scene: CupManipulationSceneIDs,
    scene_description: CupManipulationSceneDescription,
    max_motion_plan_time: float = 10.0,
    num_grasp_waypoints: int = 5,
    seed: int = 0,
    num_drink_transfer_end_effector_interp: int = 25,
    max_joint_space_distance: float = 0.1,
    force_rerun: bool = False,
) -> CupManipulationTrajectory:
    """Run planning to create a cup manipulation trajectory."""

    saved_traj_dir = Path(__file__).parent / "saved_trajs"
    os.makedirs(saved_traj_dir, exist_ok=True)
    saved_traj_files = list(saved_traj_dir.glob("*.traj"))
    if not saved_traj_files:
        next_file_id = 0
    else:
        next_file_id = len(saved_traj_files)
    next_filepath = saved_traj_dir / f"{next_file_id}.traj"
    assert not next_filepath.exists()

    # Check if we already have saved a trajectory for this scene.
    if not force_rerun:
        for saved_traj_file in saved_traj_files:
            with open(saved_traj_file, "rb") as rfp:
                saved_scene_description, saved_traj = pickle.load(rfp)
            if scene_description.allclose(saved_scene_description):
                traj = saved_traj
                print(f"Loaded saved trajectory: {saved_traj_file.name}")
                return traj

    # Need to replan.
    print("Running planning...")
    plan = CupManipulationTrajectory()

    robot = scene.robot
    physics_client_id = scene.physics_client_id
    assert robot.physics_client_id == physics_client_id

    # Create joint distance function.
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

    bite_transfer_plan = _get_move_utensil_to_transfer_plan(
        forque_target_pose,
        scene,
        scene_description,
        _joint_distance_fn,
        max_motion_plan_time,
    )
    plan.extend(bite_transfer_plan)
    robot.set_joints(plan.joint_states[-1])

    # Remap the trajectory.
    plan = _remap_trajectory_to_constant_distance(
        plan, scene, _joint_distance_fn, max_joint_space_distance
    )

    # Rajat ToDo: Fix error that occurs when trying to save the trajectory ('TypeError: cannot pickle 'generator' object)
    # # Save the trajectory.
    # with open(next_filepath, "wb") as wfp:
    #     pickle.dump((scene_description, plan), wfp)
    # print(f"Dumped trajectory to {next_filepath}")

    print("Type of plan: ", type(plan))
    input("Press enter to continue")

    return plan


def _main(seed: int, max_motion_plan_time: float, force_rerun: bool) -> None:

    scene_description = CupManipulationSceneDescription()
    physics_client_id = create_gui_connection(camera_yaw=180)
    scene = create_cup_manipulation_scene(physics_client_id, scene_description)

    traj = generate_trajectory(
        scene,
        scene_description,
        max_motion_plan_time=max_motion_plan_time,
        seed=seed,
        force_rerun=force_rerun,
    )

    video_outfile = Path("generated_trajectory.mp4")
    make_cup_manipulation_video(
        scene,
        scene_description,
        traj,
        video_outfile,
    )
    p.disconnect(physics_client_id)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_motion_plan_time", type=float, default=5.0)
    parser.add_argument("--force_rerun", action="store_true")
    args = parser.parse_args()

    _main(args.seed, args.max_motion_plan_time, args.force_rerun)
