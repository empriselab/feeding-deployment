"""Script to develop cup manipulation skills in simulation."""

from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.single_arm import (
    SingleArmPyBulletRobot,
)
from pybullet_helpers.inverse_kinematics import (
    sample_collision_free_inverse_kinematics,
    inverse_kinematics,
)
from pybullet_helpers.geometry import Pose, Pose3D, get_pose, multiply_poses
from pybullet_helpers.link import get_relative_link_pose, get_link_pose
from pybullet_helpers.joint import (
    JointPositions,
    get_joint_infos,
)
from pybullet_helpers.camera import capture_image
from pybullet_helpers.utils import create_pybullet_block
from pybullet_helpers.motion_planning import (
    run_motion_planning,
    get_joint_positions_distance,
    select_shortest_motion_plan,
    set_robot_joints_with_held_object,
)
from pybullet_helpers.gui import visualize_pose, create_gui_connection
from pybullet_helpers.geometry import matrix_from_quat
from pybullet_helpers.math_utils import get_poses_facing_line
import numpy as np
from functools import partial
from pathlib import Path
import imageio.v2 as iio
from tqdm import tqdm
from numpy.typing import NDArray
from tomsutils.structs import Image
from functools import lru_cache

import pybullet as p


def _initialize_scene(
    robot_initial_joints: list[JointPositions],
    robot_base_pose: Pose, wheelchair_pose: Pose
) -> tuple[SingleArmPyBulletRobot, Pose, int, int, set[int]]:
    """Returns robot, cup ID, table ID, other collision IDs."""

    robot_holder_rgba = (0.5, 0.5, 0.5, 1.0)
    robot_holder_half_extents = (0.25, 0.25, 0.5)
    robot_holder_position = (0.0, 0.0, -0.5)
    robot_holder_orientation = (0.0, 0.0, 0.0, 1.0)

    collision_region1_orientation = (0.0, 0.0, 0.0, 1.0)
    collision_region1_position = (-0.75, -0.4, -0.25)
    collision_region1_half_extents = (0.5, 0.5, 0.25)
    collision_region1_rgba = (1.0, 0.0, 0.0, 0.25)

    collision_region2_orientation = (0.0, 0.0, 0.0, 1.0)
    collision_region2_position = (-0.075, 0.55, 0.05)
    collision_region2_half_extents = (0.05, 0.05, 0.05)
    collision_region2_rgba = (1.0, 0.0, 0.0, 0.25)

    wheelchair_position = wheelchair_pose.position
    wheelchair_orientation = wheelchair_pose.orientation

    table_rgba = (0.5, 0.5, 0.5, 1.0)
    table_half_extents = (0.75, 0.25, 0.5)
    table_position = (-0.5, 0.75, -0.5)
    table_orientation = (0.0, 0.0, 0.0, 1.0)

    cup_rgba = (0.0, 0.0, 1.0, 1.0)
    cup_radius = 0.03
    cup_length = 0.15
    cup_position = (0.0, 0.75, cup_length / 2)
    cup_orientation = (0.0, 0.0, 0.0, 1.0)
    cup_handle_half_extents = (cup_radius, cup_radius, cup_radius)
    cup_handle_rgba = (0.7, 0.2, 0.5, 1.0)
    cup_handle_relative_position = (
        0.0,
        -cup_radius - cup_handle_half_extents[1] / 2,
        cup_length / 4,
    )
    cup_handle_relative_orientation = (0.0, 0.0, 0.0, 1.0)

    physics_client_id = create_gui_connection(camera_yaw=180)
    p.setGravity(0.0, 0.0, 0.0, physicsClientId=physics_client_id)

    # Create robot.
    robot = create_pybullet_robot(
        "kinova-gen3",
        physics_client_id,
        base_pose=robot_base_pose,
        control_mode="reset",
        home_joint_positions=robot_initial_joints,
    )

    # Create a base for visualization purposes.
    robot_holder_id = create_pybullet_block(
        robot_holder_rgba,
        half_extents=robot_holder_half_extents,
        physics_client_id=physics_client_id,
    )
    p.resetBasePositionAndOrientation(
        robot_holder_id,
        robot_holder_position,
        robot_holder_orientation,
        physicsClientId=physics_client_id,
    )

    wheelchair_urdf_path = (
        Path(__file__).parent.parent
        / "assets"
        / "urdf"
        / "wheelchair"
        / "wheelchair.urdf"
    )
    wheelchair_id = p.loadURDF(
        str(wheelchair_urdf_path), useFixedBase=True, physicsClientId=physics_client_id
    )
    p.resetBasePositionAndOrientation(
        wheelchair_id,
        wheelchair_position,
        wheelchair_orientation,
        physicsClientId=physics_client_id,
    )
    collision_region_ids = {wheelchair_id}

    # Create cup.
    cup_collision_id = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=cup_radius,
        height=cup_length,
        physicsClientId=physics_client_id,
    )
    cup_visual_id = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=cup_radius,
        length=cup_length,
        rgbaColor=cup_rgba,
        physicsClientId=physics_client_id,
    )
    cup_handle_collision_id = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=cup_handle_half_extents,
        physicsClientId=physics_client_id,
    )
    cup_handle_visual_id = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=cup_handle_half_extents,
        rgbaColor=cup_handle_rgba,
        physicsClientId=physics_client_id,
    )
    cup_id = p.createMultiBody(
        baseMass=-1,
        baseCollisionShapeIndex=cup_collision_id,
        baseVisualShapeIndex=cup_visual_id,
        basePosition=cup_position,
        baseOrientation=cup_orientation,
        linkMasses=[-1],
        linkCollisionShapeIndices=[cup_handle_collision_id],
        linkVisualShapeIndices=[cup_handle_visual_id],
        linkPositions=[cup_handle_relative_position],
        linkOrientations=[cup_handle_relative_orientation],
        linkInertialFramePositions=[(0.0, 0.0, 0.0)],
        linkInertialFrameOrientations=[(0.0, 0.0, 0.0, 1.0)],
        linkParentIndices=[0],
        linkJointTypes=[p.JOINT_FIXED],
        linkJointAxis=[(0.0, 0.0, 1.0)],
        physicsClientId=physics_client_id,
    )

    table_id = create_pybullet_block(
        table_rgba,
        half_extents=table_half_extents,
        physics_client_id=physics_client_id,
    )
    p.resetBasePositionAndOrientation(
        table_id,
        table_position,
        table_orientation,
        physicsClientId=physics_client_id,
    )

    # Create additional collision areas.

    # collision_region_id1 = create_pybullet_block(
    #     collision_region1_rgba,
    #     half_extents=collision_region1_half_extents,
    #     physics_client_id=physics_client_id,
    # )
    # p.resetBasePositionAndOrientation(
    #     collision_region_id1,
    #     collision_region1_position,
    #     collision_region1_orientation,
    #     physicsClientId=physics_client_id,
    # )
    # collision_region_ids.add(collision_region_id1)

    # collision_region_id2 = create_pybullet_block(
    #     collision_region2_rgba,
    #     half_extents=collision_region2_half_extents,
    #     physics_client_id=physics_client_id,
    # )
    # p.resetBasePositionAndOrientation(
    #     collision_region_id2,
    #     collision_region2_position,
    #     collision_region2_orientation,
    #     physicsClientId=physics_client_id,
    # )
    # collision_region_ids.add(collision_region_id2)

    return robot, cup_id, table_id, collision_region_ids


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


def _select_smoothest_motion_plan(
    robot: SingleArmPyBulletRobot,
    motion_plans: list[list[JointPositions]],
    joint_geometric_scalar: float = 0.9,
) -> float:
    """Lower is better."""

    # Geometric weighting so the base moves less than the end effector, etc.
    weights = [1.0]
    num_joints = len(robot.arm_joints)
    for _ in range(num_joints - 1):
        weights.append(weights[-1] * joint_geometric_scalar)

    joint_infos = get_joint_infos(
        robot.robot_id, robot.arm_joints, robot.physics_client_id
    )
    dist_fn = partial(
        get_joint_positions_distance,
        robot,
        joint_infos,
        metric="weighted_joints",
        weights=weights,
    )

    return select_shortest_motion_plan(motion_plans, dist_fn)


def _move_end_effector(robot: SingleArmPyBulletRobot, tf: Pose) -> None:
    # TODO: put back into pybullet-helpers
    current_end_effector_pose = robot.get_end_effector_pose()
    next_end_effector_pose = multiply_poses(current_end_effector_pose, tf)
    return inverse_kinematics(robot, next_end_effector_pose, set_joints=False)


def _smooth_motion_plan(
    target_poses: list[Pose],
    robot: SingleArmPyBulletRobot,
    collision_ids: set[int],
    plan_frame_from_end_effector_frame: Pose,
    seed: int,
    held_object: int | None = None,
    base_link_to_held_obj: NDArray | None = None,
    max_ik_candidates_per_target_pose: int = 100,
) -> list[JointPositions]:

    robot_initial_joints = robot.get_joint_positions()

    # Find a number of possible target joint positions.
    all_target_joint_positions = []
    for target_pose in target_poses:
        end_effector_pose = multiply_poses(
            target_pose, plan_frame_from_end_effector_frame
        )
        for candidate_joints in sample_collision_free_inverse_kinematics(
            robot,
            end_effector_pose,
            collision_ids,
            max_candidates=max_ik_candidates_per_target_pose,
        ):
            robot.set_joints(candidate_joints)
            all_target_joint_positions.append(candidate_joints)

    print(f"Found {len(all_target_joint_positions)} candidate joint positions.")

    # Motion plan to each.
    print("Starting motion planning...")
    all_motion_plans = []
    for target_joint_positions in tqdm(all_target_joint_positions):
        robot.set_joints(robot_initial_joints)
        plan = run_motion_planning(
            robot,
            robot_initial_joints,
            target_joint_positions,
            collision_ids,
            seed,
            robot.physics_client_id,
            held_object=held_object,
            base_link_to_held_obj=base_link_to_held_obj,
        )
        if plan is not None:
            all_motion_plans.append(plan)

    print(f"Found {len(all_motion_plans)} motion plans.")

    # Choose the best motion plan.
    plan = _select_smoothest_motion_plan(robot, all_motion_plans)

    return plan


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
    robot_initial_joints: list[JointPositions],
    robot_base_pose: Pose = Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    wheelchair_pose: Pose = Pose((-0.5, 0.0, -0.25), (0.0, 0.0, 1.0, 0.0)),
    wheelchair_relative_head_pose: Pose = Pose(
        (0.0, -0.25, 0.75), (0.0, 0.0, 0.0, 1.0)
    ),
    pregrasp_distance: float = 0.075,
    max_num_grasps: int = 25,
    num_grasp_waypoints: int = 5,
    staging_relative_pose=Pose(
        (-0.1, 0.5, 0.0), p.getQuaternionFromEuler((0.0, 0.0, np.pi / 2))
    ),
    seed: int = 0,
    make_video: bool = True,
) -> list[JointPositions]:

    wheelchair_center_pose = Pose(wheelchair_pose.position, robot_base_pose.orientation)
    wheelchair_head_pose = multiply_poses(
        wheelchair_center_pose, wheelchair_relative_head_pose
    )

    robot, cup_id, table_id, other_collision_ids = _initialize_scene(
        robot_initial_joints, robot_base_pose, wheelchair_pose
    )
    collision_ids = {cup_id, table_id} | other_collision_ids
    physics_client_id = robot.physics_client_id
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
    table_frame = get_pose(table_id, physics_client_id)
    # Move the frame to the bottom right hand corner of the table so we can see it.
    dims = p.getVisualShapeData(table_id, physicsClientId=physics_client_id)[0][3]
    offset = Pose((dims[0] / 2, -dims[1] / 2, dims[2] / 2))
    table_frame = multiply_poses(offset, table_frame)

    visualize_pose(table_frame, physics_client_id)

    # Find target finger frame pose relative to the cup handle.
    cup_handle_link_id = 0
    cup_pose = get_link_pose(cup_id, cup_handle_link_id, physics_client_id)

    rng = np.random.default_rng(seed)
    target_poses = [
        _sample_grasp(cup_pose, rng, pregrasp_distance) for _ in range(max_num_grasps)
    ]
    plan = _smooth_motion_plan(
        target_poses,
        robot,
        collision_ids,
        finger_from_end_effector,
        seed,
    )

    # Execute the motion plan.
    imgs = []
    for state in plan:
        robot.set_joints(state)
        all_joint_positions.append(state)
        if make_video:
            img = _capture_image(
                physics_client_id,
                robot_base_pose.position,
                wheelchair_head_pose.position,
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
                robot_base_pose.position,
                wheelchair_head_pose.position,
            )
            imgs.append(img)

    # Open the fingers to create a constraint inside the mounted holder.
    robot.open_fingers()
    all_joint_positions.append(robot.get_joint_positions())

    # Simulate grasping by faking a constraint with the held object.
    held_obj_id = cup_id
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
            robot_base_pose.position,
            wheelchair_head_pose.position,
        )
        imgs.append(img)

    # Move to staging pose.
    new_cup_pose = multiply_poses(wheelchair_head_pose, staging_relative_pose)
    fingers_to_cup = multiply_poses(
        cup_pose.invert(),
        get_link_pose(robot.robot_id, finger_frame_id, physics_client_id),
    )
    new_fingers_pose = multiply_poses(new_cup_pose, fingers_to_cup)
    visualize_pose(new_fingers_pose, physics_client_id)

    new_collision_ids = collision_ids - {held_obj_id}
    plan = _smooth_motion_plan(
        [new_fingers_pose],
        robot,
        new_collision_ids,
        finger_from_end_effector,
        seed,
        held_object=held_obj_id,
        base_link_to_held_obj=base_link_to_held_obj,
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
                robot_base_pose.position,
                wheelchair_head_pose.position,
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
                robot_base_pose.position,
                wheelchair_head_pose.position,
            )
            imgs.append(img)
        iio.mimsave("replayed_trajectory.mp4", imgs, fps=20)

    p.disconnect(physics_client_id)

    return all_joint_positions


if __name__ == "__main__":
    robot_initial_joints = [
        np.pi / 2,
        -np.pi / 4,
        -np.pi / 2,
        0.0,
        np.pi / 2,
        -np.pi / 2,
        np.pi / 2,
        0.0,
        0.0,
    ]
    all_joint_positions = generate_trajectory(robot_initial_joints)
