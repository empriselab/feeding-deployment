"""Script to develop cup manipulation skills in simulation."""

from pybullet_helpers.inverse_kinematics import (
    set_robot_joints_with_held_object,
    end_effector_transform_to_joints,
    sample_joints_from_task_space_bounds,
)
from pybullet_helpers.geometry import Pose, Pose3D, get_pose, multiply_poses
from pybullet_helpers.link import get_relative_link_pose, get_link_pose
from pybullet_helpers.joint import (
    JointPositions,
)
from pybullet_helpers.motion_planning import (
    run_smooth_motion_planning_to_pose,
)
from pybullet_helpers.gui import visualize_pose, create_gui_connection
from functools import lru_cache
from pathlib import Path

from scene import (
    create_cup_manipulation_scene,
    CupManipulationSceneIDs,
    CupManipulationSceneDescription,
)
from utils import make_cup_manipulation_video, CupManipulationTrajectory

import pybullet as p
import numpy as np
import pickle


@lru_cache(maxsize=None)
def generate_trajectory(
    scene: CupManipulationSceneIDs,
    scene_description: CupManipulationSceneDescription,
    max_motion_plan_time: int = 10,
    num_grasp_waypoints: int = 5,
    seed: int = 0,
) -> CupManipulationTrajectory:

    physics_client_id = scene.physics_client_id
    robot = scene.robot

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
    move_amt = scene_description.cup_grasp_distance / (num_grasp_waypoints - 1)
    tf = Pose(
        (0.0, 0.0, move_amt),
        (0.0, 0.0, 0.0, 1.0),
    )
    for _ in range(num_grasp_waypoints):
        joints = end_effector_transform_to_joints(robot, tf)
        robot.set_joints(joints)
        all_joint_positions.append(joints)
        all_held_cup_tfs.append(None)

    # Open the fingers to create a constraint inside the mounted holder.
    robot.open_fingers()
    all_joint_positions.append(robot.get_joint_positions())
    all_held_cup_tfs.append(None)

    # Simulate grasping by faking a constraint with the held object.
    held_obj_id = scene.cup_id
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
    new_cup_pose = multiply_poses(
        scene_description.wheelchair_head_pose, scene_description.staging_relative_pose
    )
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

    # Prevent spilling.
    rng = np.random.default_rng(seed)
    # Determine reasonable task space bounds by drawing a box around the end
    # effector and the target and then expanding it by quite a bit.
    xs = [current_fingers_pose.position[0], new_fingers_pose.position[0]]
    ys = [current_fingers_pose.position[1], new_fingers_pose.position[1]]
    zs = [current_fingers_pose.position[2], new_fingers_pose.position[2]]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)
    scale_x, scale_y, scale_z = max_x - min_x, max_y - min_y, max_z - max_z
    scale_factor = 1.5
    min_x -= scale_factor * scale_x
    max_x += scale_factor * scale_x
    min_y -= scale_factor * scale_y
    max_y += scale_factor * scale_y
    min_z -= scale_factor * scale_z
    max_z += scale_factor * scale_z
    # Constrain the roll to be close to the current.
    current_roll = p.getEulerFromQuaternion(current_fingers_pose.orientation)[0]
    new_roll = p.getEulerFromQuaternion(new_fingers_pose.orientation)[0]
    assert np.isclose(current_roll, new_roll)
    min_roll = current_roll - 1e-6
    max_roll = current_roll + 1e-6
    min_pitch, max_pitch = -np.pi, np.pi
    min_yaw, max_yaw = -np.pi, np.pi

    def _sample_fn(_current_joint_positions: JointPositions) -> JointPositions:
        del _current_joint_positions  # not used
        return sample_joints_from_task_space_bounds(
            rng,
            robot,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            min_z=min_z,
            max_z=max_z,
            min_roll=min_roll,
            max_roll=max_roll,
            min_pitch=min_pitch,
            max_pitch=max_pitch,
            min_yaw=min_yaw,
            max_yaw=max_yaw,
        )

    new_collision_ids = collision_ids - {held_obj_id}
    plan = run_smooth_motion_planning_to_pose(
        new_fingers_pose,
        robot,
        new_collision_ids,
        finger_from_end_effector,
        seed,
        held_object=held_obj_id,
        base_link_to_held_obj=base_link_to_held_obj,
        max_time=max_motion_plan_time,
        sampling_fn=_sample_fn,
    )

    # Execute the motion plan.
    for state in plan:
        set_robot_joints_with_held_object(
            robot, physics_client_id, held_obj_id, base_link_to_held_obj, state
        )
        all_joint_positions.append(state)
        all_held_cup_tfs.append(base_link_to_held_obj)

    return CupManipulationTrajectory(all_joint_positions, all_held_cup_tfs)


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
