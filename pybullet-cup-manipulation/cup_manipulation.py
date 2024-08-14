"""Script to develop cup manipulation skills in simulation."""

from pybullet_helpers.robots.single_arm import (
    SingleArmPyBulletRobot,
)
from pybullet_helpers.inverse_kinematics import (
    inverse_kinematics,
    set_robot_joints_with_held_object,
)
from pybullet_helpers.geometry import Pose, Pose3D, get_pose, multiply_poses
from pybullet_helpers.link import get_relative_link_pose, get_link_pose
from pybullet_helpers.joint import (
    JointPositions,
)
from pybullet_helpers.camera import capture_image
from pybullet_helpers.motion_planning import (
    run_smooth_motion_planning_to_pose,
)
from pybullet_helpers.gui import visualize_pose, create_gui_connection
from pybullet_helpers.geometry import matrix_from_quat
from pybullet_helpers.math_utils import get_poses_facing_line
import numpy as np
import imageio.v2 as iio
from tomsutils.structs import Image
from functools import lru_cache

from scene import create_cup_manipulation_scene, CupManipulationSceneDescription

import pybullet as p


def _sample_grasp(
    cup_pose: Pose, rng: np.random.Generator, translation_distance: float = 0.2
) -> Pose:

    cup_point = cup_pose.position
    cup_matrix = matrix_from_quat(cup_pose.orientation)
    cup_z_axis = cup_matrix[:, 2]
    angle_offset = rng.uniform(-np.pi, np.pi)

    grasp_pose = get_poses_facing_line(
        cup_z_axis, cup_point, translation_distance, 1, angle_offset=angle_offset
    )[0]

    return grasp_pose


def _move_end_effector(robot: SingleArmPyBulletRobot, tf: Pose) -> None:
    # TODO: put back into pybullet-helpers
    current_end_effector_pose = robot.get_end_effector_pose()
    next_end_effector_pose = multiply_poses(current_end_effector_pose, tf)
    return inverse_kinematics(robot, next_end_effector_pose, set_joints=False)


def _capture_image(
    physics_client_id: int,
    base_position: Pose3D,
    head_position: Pose3D,
) -> Image:

    outer_size = 900
    inner_size = outer_size // 3
    pad_size = outer_size // 50
    border_size = pad_size // 3

    outer_image = capture_image(
        physics_client_id,
        camera_target=base_position,
        camera_yaw=180,
        camera_distance=2.5,
        camera_pitch=-20,
        image_height=outer_size,
        image_width=outer_size,
    )

    inner_image = capture_image(
        physics_client_id,
        camera_target=head_position,
        camera_yaw=0,
        camera_distance=1.0,
        camera_pitch=-20,
        image_height=inner_size,
        image_width=inner_size,
    )

    combined_image = outer_image.copy()

    frame_size = pad_size + border_size
    combined_image[
        pad_size : inner_size + frame_size, pad_size : inner_size + frame_size
    ] = 200.0

    combined_image[
        pad_size : inner_size + pad_size, pad_size : inner_size + pad_size
    ] = inner_image

    return combined_image.astype(np.uint8)


@lru_cache(maxsize=None)
def generate_trajectory(
    scene_description: CupManipulationSceneDescription,
    pregrasp_distance: float = 0.075,
    max_motion_plan_time: int = 10,
    num_grasp_waypoints: int = 5,
    seed: int = 0,
    make_video: bool = True,
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

    # Close the fingers.
    robot.close_fingers()
    all_joint_positions.append(robot.get_joint_positions())

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
    cup_handle_link_id = 0
    cup_pose = get_link_pose(scene.cup_id, cup_handle_link_id, physics_client_id)

    rng = np.random.default_rng(seed)
    pose_sampler = lambda: _sample_grasp(cup_pose, rng, pregrasp_distance)
    plan = run_smooth_motion_planning_to_pose(
        pose_sampler,
        robot,
        collision_ids,
        finger_from_end_effector,
        seed,
        max_time=max_motion_plan_time,
    )

    # Execute the motion plan.
    imgs = []
    for state in plan:
        robot.set_joints(state)
        all_joint_positions.append(state)
        if make_video:
            img = _capture_image(
                physics_client_id,
                scene_description.robot_base_pose.position,
                scene_description.wheelchair_head_pose.position,
            )
            imgs.append(img)

    # Move to grasp.
    tf = Pose(
        (0.0, 0.0, pregrasp_distance / (num_grasp_waypoints - 1)), (0.0, 0.0, 0.0, 1.0)
    )
    for _ in range(num_grasp_waypoints):
        joints = _move_end_effector(robot, tf)
        robot.set_joints(joints)
        all_joint_positions.append(joints)
        if make_video:
            img = _capture_image(
                physics_client_id,
                scene_description.robot_base_pose.position,
                scene_description.wheelchair_head_pose.position,
            )
            imgs.append(img)

    # Open the fingers to create a constraint inside the mounted holder.
    robot.open_fingers()
    all_joint_positions.append(robot.get_joint_positions())

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
    joints = _move_end_effector(robot, tf)
    set_robot_joints_with_held_object(
        robot, physics_client_id, held_obj_id, base_link_to_held_obj, joints
    )
    all_joint_positions.append(joints)
    if make_video:
        img = _capture_image(
            physics_client_id,
            scene_description.robot_base_pose.position,
            scene_description.wheelchair_head_pose.position,
        )
        imgs.append(img)

    # Move to staging pose.
    new_cup_pose = multiply_poses(
        scene_description.wheelchair_head_pose, scene_description.staging_relative_pose
    )
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
        if make_video:
            img = _capture_image(
                physics_client_id,
                scene_description.robot_base_pose.position,
                scene_description.wheelchair_head_pose.position,
            )
            imgs.append(img)

    if make_video:
        iio.mimsave("generated_trajectory.mp4", imgs, fps=20)

        # Replay the whole trajectory.
        imgs = []
        for state in all_joint_positions:
            robot.set_joints(state)
            img = _capture_image(
                physics_client_id,
                scene_description.robot_base_pose.position,
                scene_description.wheelchair_head_pose.position,
            )
            imgs.append(img)
        iio.mimsave("replayed_trajectory.mp4", imgs, fps=20)

    p.disconnect(physics_client_id)

    return all_joint_positions


if __name__ == "__main__":
    scene_description = CupManipulationSceneDescription()

    # scene_rotation = tuple(
    #     p.getQuaternionFromEuler((np.pi / 8, -np.pi / 4, -np.pi / 2))
    # )
    # scene_description = scene_description.rotate_about_point(
    #     (0.0, 0.0, 0.0), scene_rotation
    # )

    generate_trajectory(scene_description)

    # from scipy.spatial.transform import Rotation

    # quat = tuple(Rotation.from_euler('xyz', [0, 0, 90], degrees=True).as_quat())
    # robot_base_pose = Pose((0.0, 0.0, 0.0), quat)
    # robot_initial_joints = (2.80374963034063, 5.737339549099201, 3.3692055751078134, 2.2763574480856223, 3.002470982456817, 1.2413268146451608, 1.505290153988054, 0.5973799438476562, 0.5973799438476562)
    # all_joint_positions = generate_trajectory(robot_initial_joints, robot_base_pose)
