"""Script to develop cup manipulation skills in simulation."""

from pybullet_helpers.inverse_kinematics import (
    set_robot_joints_with_held_object,
    end_effector_transform_to_joints,
)
from pybullet_helpers.geometry import Pose, get_pose, multiply_poses, interpolate_poses
from pybullet_helpers.link import get_relative_link_pose, get_link_pose
from pybullet_helpers.joint import get_joint_infos, interpolate_joints
from pybullet_helpers.motion_planning import (
    run_smooth_motion_planning_to_pose,
    smoothly_follow_end_effector_path,
    get_joint_positions_distance,
)
from pybullet_helpers.math_utils import geometric_sequence
from pybullet_helpers.gui import visualize_pose, create_gui_connection
from pybullet_helpers.trajectory import (
    concatenate_trajectories,
    TrajectorySegment,
    iter_traj_with_max_distance,
)

from functools import partial
from pathlib import Path

from scene import (
    create_cup_manipulation_scene,
    CupManipulationSceneIDs,
    CupManipulationSceneDescription,
)
from cup_manipulation_utils import (
    make_cup_manipulation_video,
    CupManipulationTrajectory,
)

import pybullet as p
import numpy as np
import pickle


def generate_trajectory(
    scene: CupManipulationSceneIDs,
    scene_description: CupManipulationSceneDescription,
    max_motion_plan_time: int = 10,
    num_grasp_waypoints: int = 5,
    seed: int = 0,
    num_drink_transfer_end_effector_interp: int = 25,
    max_joint_space_distance=0.1,
) -> CupManipulationTrajectory:

    physics_client_id = scene.physics_client_id
    robot = scene.robot

    weights = geometric_sequence(0.9, len(robot.arm_joint_names))
    joint_infos = get_joint_infos(
        robot.robot_id, robot.arm_joints, robot.physics_client_id
    )

    def joint_distance_fn(pt1, pt2):
        return get_joint_positions_distance(
            robot,
            joint_infos,
            pt1,
            pt2,
            metric="weighted_joints",
            weights=weights,
        )

    collision_ids = {
        scene.cup_id,
        scene.table_id,
        scene.robot_holder_id,
        scene.wheelchair_id,
    }
    all_joint_positions = [robot.get_joint_positions()]
    all_held_cup_tfs = [None]

    # Close the fingers.
    robot.close_fingers()
    all_joint_positions.append(robot.get_joint_positions())
    all_held_cup_tfs.append(None)

    # Commands will be in end effector space, but grasp planning will be in
    # finger frame space.
    finger_frame_id = robot.link_from_name("finger_tip")
    end_effector_link_id = robot.link_from_name(robot.tool_link_name)
    finger_from_end_effector = get_relative_link_pose(
        robot.robot_id, end_effector_link_id, finger_frame_id, physics_client_id
    )

    # Use the table pose as a frame of reference.
    table_frame = get_pose(scene.table_id, physics_client_id)
    # Move the frame to the bottom right hand corner of the table so we can see it.
    dims = p.getVisualShapeData(scene.table_id, physicsClientId=physics_client_id)[0][3]
    offset = Pose((dims[0] / 2, -dims[1] / 2, dims[2] / 2))
    table_frame = multiply_poses(table_frame, offset)

    visualize_pose(table_frame, physics_client_id)

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

    # Execute the motion plan.
    for state in plan:
        robot.set_joints(state)
        all_joint_positions.append(state)
        all_held_cup_tfs.append(None)

    # Move to grasp.
    held_obj_id = scene.cup_id
    new_collision_ids = collision_ids - {held_obj_id}
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
        new_collision_ids,
        joint_distance_fn,
        max_time=max_motion_plan_time,
    )

    # Execute the plan.
    for state in plan:
        robot.set_joints(state)
        all_joint_positions.append(state)
        all_held_cup_tfs.append(None)

    # Open the fingers to create a constraint inside the mounted holder.
    robot.open_fingers()
    all_joint_positions.append(robot.get_joint_positions())
    all_held_cup_tfs.append(None)

    # Simulate grasping by faking a constraint with the held object.
    world_from_end_effector = get_link_pose(
        robot.robot_id, robot.end_effector_id, physics_client_id
    )
    world_from_held_object = get_pose(held_obj_id, physics_client_id)
    base_link_to_held_obj = multiply_poses(
        world_from_end_effector.invert(), world_from_held_object
    )

    # Move off the table so that the cup is no longer in collision with the table.
    tf = Pose((0.0, -0.01, 0.0), (0.0, 0.0, 0.0, 1.0))
    joints = end_effector_transform_to_joints(robot, tf)
    set_robot_joints_with_held_object(
        robot, physics_client_id, held_obj_id, base_link_to_held_obj, joints
    )
    all_joint_positions.append(joints)
    all_held_cup_tfs.append(base_link_to_held_obj)

    # Move to staging pose.
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
    visualize_pose(new_fingers_pose, physics_client_id)
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
        new_collision_ids,
        joint_distance_fn,
        max_time=max_motion_plan_time,
    )

    # Execute the plan.
    for state in plan:
        set_robot_joints_with_held_object(
            robot, physics_client_id, held_obj_id, base_link_to_held_obj, state
        )
        all_joint_positions.append(state)
        all_held_cup_tfs.append(base_link_to_held_obj)

    # Try to move closer to the mouth for bite transfer.
    # transfer_relative_pose: Pose = Pose(
    #     (-0.5, 0.1, 0.0), p.getQuaternionFromEuler((0.0, 0.0, np.pi / 2))
    # )
    # cup_transfer_pose = multiply_poses(scene_description.wheelchair_head_pose, transfer_relative_pose)
    # current_end_effector_pose = robot.get_end_effector_pose()
    # new_fingers_pose = multiply_poses(cup_transfer_pose, fingers_to_cup)
    # visualize_pose(new_fingers_pose, physics_client_id)

    # new_end_effector_pose = multiply_poses(new_fingers_pose, finger_from_end_effector)
    # interpolated_poses = list(
    #     interpolate_poses(
    #         current_end_effector_pose,
    #         new_end_effector_pose,
    #         num_interp=10,  # NOTE
    #     )
    # )
    # plan = smoothly_follow_end_effector_path(
    #     robot,
    #     interpolated_poses,
    #     robot.get_joint_positions(),
    #     new_collision_ids,
    #     joint_distance_fn,
    #     max_time=max_motion_plan_time,
    # )
    # # Execute the plan.
    # for state in plan:
    #     set_robot_joints_with_held_object(
    #         robot, physics_client_id, held_obj_id, base_link_to_held_obj, state
    #     )
    #     all_joint_positions.append(state)
    #     all_held_cup_tfs.append(base_link_to_held_obj)

    # Create a continuous-time trajectory.
    joint_interpolate_fn = partial(interpolate_joints, joint_infos)
    distances = []
    for pt1, pt2 in zip(all_joint_positions[:-1], all_joint_positions[1:], strict=True):
        dist = joint_distance_fn(pt1, pt2)
        distances.append(dist)
    # Use distances as times.
    segments = []
    for t in range(len(all_joint_positions) - 1):
        start = all_joint_positions[t]
        end = all_joint_positions[t + 1]
        dt = distances[t]
        seg = TrajectorySegment(
            start,
            end,
            dt,
            interpolate_fn=joint_interpolate_fn,
            distance_fn=joint_distance_fn,
        )
        segments.append(seg)
    continuous_time_trajectory = concatenate_trajectories(segments)
    remapped_joint_positions = list(
        iter_traj_with_max_distance(
            continuous_time_trajectory, max_joint_space_distance
        )
    )

    # Remap the cup states.
    def cup_interpolate_fn(q1, q2, t):
        return q1

    def cup_distance_fn(q1, q2):
        raise NotImplementedError

    cup_segments = []
    for t in range(len(all_held_cup_tfs) - 1):
        start = all_held_cup_tfs[t]
        end = all_held_cup_tfs[t + 1]
        dt = distances[t]
        seg = TrajectorySegment(
            start,
            end,
            dt,
            interpolate_fn=cup_interpolate_fn,
            distance_fn=cup_distance_fn,
        )
        cup_segments.append(seg)
    continuous_time_cup_trajectory = concatenate_trajectories(cup_segments)

    ts = np.linspace(
        0,
        continuous_time_trajectory.duration,
        num=len(remapped_joint_positions),
        endpoint=True,
    )
    remapped_held_cup_tfs = [continuous_time_cup_trajectory(t) for t in ts]

    return CupManipulationTrajectory(remapped_joint_positions, remapped_held_cup_tfs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_motion_plan_time", type=float, default=5.0)
    parser.add_argument("--force_rerun", action="store_true")
    args = parser.parse_args()

    scene_description = CupManipulationSceneDescription()
    physics_client_id = create_gui_connection(camera_yaw=180)
    scene = create_cup_manipulation_scene(physics_client_id, scene_description)

    saved_traj_dir = Path(__file__).parent / "saved_trajs"
    assert saved_traj_dir.exists()
    saved_traj_files = list(saved_traj_dir.glob("*.traj"))
    if not saved_traj_files:
        next_file_id = 0
    else:
        next_file_id = len(saved_traj_files)
    next_filepath = saved_traj_dir / f"{next_file_id}.traj"
    assert not next_filepath.exists()

    # Check if we already have saved a trajectory for this scene.
    traj = None
    if not args.force_rerun:
        for saved_traj_file in saved_traj_files:
            with open(saved_traj_file, "rb") as f:
                saved_scene_description, saved_traj = pickle.load(f)
            if scene_description.allclose(saved_scene_description):
                traj = saved_traj
                print(f"Loaded saved trajectory: {saved_traj_file.name}")
                break

    # Need to replan.
    if traj is None:
        traj = generate_trajectory(
            scene,
            scene_description,
            seed=args.seed,
            max_motion_plan_time=args.max_motion_plan_time,
        )
        with open(next_filepath, "wb") as f:
            pickle.dump((scene_description, traj), f)
        print(f"Dumped trajectory to {next_filepath}")

    video_outfile = Path("generated_trajectory.mp4")
    make_cup_manipulation_video(
        scene,
        scene_description,
        traj,
        video_outfile,
    )
    p.disconnect(physics_client_id)
