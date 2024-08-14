"""Script to develop cup manipulation skills in simulation."""

from pybullet_helpers.inverse_kinematics import (
    set_robot_joints_with_held_object,
    end_effector_transform_to_joints,
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
    CupManipulationSceneDescription,
)
from utils import make_cup_manipulation_video

import pybullet as p


@lru_cache(maxsize=None)
def generate_trajectory(
    scene_description: CupManipulationSceneDescription,
    max_motion_plan_time: int = 10,
    num_grasp_waypoints: int = 5,
    seed: int = 0,
    make_video: bool = True,
    video_outfile: Path = Path("generated_trajectory.mp4"),
) -> list[JointPositions]:

    physics_client_id = create_gui_connection(camera_yaw=180)
    scene = create_cup_manipulation_scene(physics_client_id, scene_description)
    robot = scene.robot

    collision_ids = {
        scene.cup_id,
        scene.table_id,
        scene.robot_holder_id,
        scene.wheelchair_id,
    }
    all_joint_positions = [robot.get_joint_positions()]
    all_held_object_infos = [None]

    # Close the fingers.
    robot.close_fingers()
    all_joint_positions.append(robot.get_joint_positions())
    all_held_object_infos.append(None)

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
        all_held_object_infos.append(None)

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
        all_held_object_infos.append(None)

    # Open the fingers to create a constraint inside the mounted holder.
    robot.open_fingers()
    all_joint_positions.append(robot.get_joint_positions())
    all_held_object_infos.append(None)

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
    all_held_object_infos.append((held_obj_id, base_link_to_held_obj))

    # Move to staging pose.
    new_cup_pose = multiply_poses(
        scene_description.wheelchair_head_pose, scene_description.staging_relative_pose
    )
    cup_pose = get_pose(scene.cup_id, physics_client_id)
    fingers_to_cup = multiply_poses(
        cup_pose.invert(),
        get_link_pose(robot.robot_id, finger_frame_id, physics_client_id),
    )
    new_fingers_pose = multiply_poses(new_cup_pose, fingers_to_cup)
    visualize_pose(new_fingers_pose, physics_client_id)

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
    )

    # Execute the motion plan.
    for state in plan:
        set_robot_joints_with_held_object(
            robot, physics_client_id, held_obj_id, base_link_to_held_obj, state
        )
        all_joint_positions.append(state)
        all_held_object_infos.append((held_obj_id, base_link_to_held_obj))

    if make_video:
        make_cup_manipulation_video(
            scene,
            scene_description,
            all_joint_positions,
            all_held_object_infos,
            video_outfile,
        )

    p.disconnect(physics_client_id)

    return all_joint_positions, all_held_object_infos


if __name__ == "__main__":
    scene_description = CupManipulationSceneDescription()

    # scene_rotation = tuple(
    #     p.getQuaternionFromEuler((np.pi / 8, -np.pi / 4, -np.pi / 2))
    # )
    # scene_description = scene_description.rotate_about_point(
    #     (0.0, 0.0, 0.0), scene_rotation
    # )

    generate_trajectory(scene_description, max_motion_plan_time=1)

    # from scipy.spatial.transform import Rotation

    # quat = tuple(Rotation.from_euler('xyz', [0, 0, 90], degrees=True).as_quat())
    # robot_base_pose = Pose((0.0, 0.0, 0.0), quat)
    # robot_initial_joints = (2.80374963034063, 5.737339549099201, 3.3692055751078134, 2.2763574480856223, 3.002470982456817, 1.2413268146451608, 1.505290153988054, 0.5973799438476562, 0.5973799438476562)
    # all_joint_positions = generate_trajectory(robot_initial_joints, robot_base_pose)
