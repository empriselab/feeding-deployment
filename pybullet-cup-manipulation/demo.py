"""Script to develop cup manipulation skills in simulation."""

from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.single_arm import (
    SingleArmPyBulletRobot,
)
from pybullet_helpers.inverse_kinematics import (
    sample_collision_free_inverse_kinematics,
    inverse_kinematics,
)
from pybullet_helpers.geometry import Pose, get_pose, multiply_poses
from pybullet_helpers.link import get_relative_link_pose, get_link_pose
from pybullet_helpers.joint import (
    JointPositions,
    get_joint_infos,
)
from pybullet_helpers.camera import capture_image
from pybullet_helpers.utils import create_pybullet_cylinder, create_pybullet_block
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

import pybullet as p


def _initialize_scene() -> tuple[SingleArmPyBulletRobot, Pose, int, int, set[int]]:
    """Returns robot, cup ID, table ID, other collision IDs."""

    robot_base_pose = Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    robot_home_joints = [
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

    robot_holder_rgba = (0.5, 0.5, 0.5, 1.0)
    robot_holder_half_extents = (0.25, 0.25, 0.5)
    robot_holder_position = (0.0, 0.0, -0.5)
    robot_holder_orientation = (0.0, 0.0, 0.0, 1.0)

    collision_region1_orientation = (0.0, 0.0, 0.0, 1.0)
    collision_region1_position = (-0.75, -0.3, 0.0)
    collision_region1_half_extents = (0.5, 0.75, 1.0)
    collision_region1_rgba = (1.0, 0.0, 0.0, 0.25)

    collision_region2_orientation = (0.0, 0.0, 0.0, 1.0)
    collision_region2_position = (-0.075, 0.55, 0.05)
    collision_region2_half_extents = (0.05, 0.05, 0.05)
    collision_region2_rgba = (1.0, 0.0, 0.0, 0.25)

    wheelchair_position = (-0.5, 0.0, -0.25)
    wheelchair_orientation = (0.0, 0.0, 1.0, 0.0)

    table_rgba = (0.5, 0.5, 0.5, 1.0)
    table_half_extents = (0.75, 0.25, 0.5)
    table_position = (-0.5, 0.75, -0.5)
    table_orientation = (0.0, 0.0, 0.0, 1.0)

    cup_rgba = (0.0, 0.0, 1.0, 1.0)
    cup_radius = 0.02
    cup_length = 0.12
    cup_position = (0.0, 0.75, cup_length / 2)
    cup_orientation = (0.0, 0.0, 0.0, 1.0)

    physics_client_id = create_gui_connection(camera_yaw=180)
    p.setGravity(0.0, 0.0, 0.0, physicsClientId=physics_client_id)

    # Create robot.
    robot = create_pybullet_robot(
        "kinova-gen3",
        physics_client_id,
        base_pose=robot_base_pose,
        control_mode="reset",
        home_joint_positions=robot_home_joints,
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

    # Create wheelchair for visualization purposes only.
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

    # Create cup.
    cup_id = create_pybullet_cylinder(
        cup_rgba,
        radius=cup_radius,
        length=cup_length,
        physics_client_id=physics_client_id,
    )
    p.resetBasePositionAndOrientation(
        cup_id,
        cup_position,
        cup_orientation,
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

    # Create collision areas.
    collision_region_ids = set()

    collision_region_id1 = create_pybullet_block(
        collision_region1_rgba,
        half_extents=collision_region1_half_extents,
        physics_client_id=physics_client_id,
    )
    p.resetBasePositionAndOrientation(
        collision_region_id1,
        collision_region1_position,
        collision_region1_orientation,
        physicsClientId=physics_client_id,
    )
    collision_region_ids.add(collision_region_id1)

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
    inverse_kinematics(robot, next_end_effector_pose, set_joints=True)


def _smooth_motion_plan(
    target_poses: list[Pose],
    robot: SingleArmPyBulletRobot,
    collision_ids: set[int],
    plan_frame_from_end_effector_frame: Pose,
    seed: int,
    held_object: int | None = None,
    base_link_to_held_obj: NDArray | None = None,
    max_ik_candidates_per_target_pose: int = 5,
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


def _main():
    seed = 0
    pregrasp_distance = 0.05

    robot, cup_id, table_id, other_collision_ids = _initialize_scene()
    collision_ids = {cup_id, table_id} | other_collision_ids
    physics_client_id = robot.physics_client_id
    robot_initial_joints = robot.get_joint_positions()

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

    # Find target finger frame pose relative to the cup.
    cup_pose = get_pose(cup_id, physics_client_id)

    rng = np.random.default_rng(seed)
    max_num_grasps = 5
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
        img = capture_image(
            physics_client_id,
            camera_yaw=180,
            camera_distance=2.5,
            camera_pitch=-20,
            image_width=900,
        )
        imgs.append(img)

    # Move to grasp.
    num_waypoints = 5
    tf = Pose((0.0, 0.0, pregrasp_distance / (num_waypoints - 1)), (0.0, 0.0, 0.0, 1.0))
    for _ in range(num_waypoints):
        _move_end_effector(robot, tf)
        visualize_pose(
            get_link_pose(robot.robot_id, finger_frame_id, physics_client_id),
            physics_client_id,
        )
        img = capture_image(
            physics_client_id,
            camera_yaw=180,
            camera_distance=2.5,
            camera_pitch=-20,
            image_width=900,
        )
        imgs.append(img)

    # Simulate grasping by faking a constraint with the held object.
    # TODO move this into pybullet-utils.
    held_obj_id = cup_id
    base_link_to_world = np.r_[
        p.invertTransform(
            *p.getLinkState(
                robot.robot_id, robot.end_effector_id, physicsClientId=physics_client_id
            )[:2]
        )
    ]
    world_to_obj = np.r_[
        p.getBasePositionAndOrientation(held_obj_id, physicsClientId=physics_client_id)
    ]
    # TODO can we combine these two lines?
    held_obj_to_base_link = p.invertTransform(
        *p.multiplyTransforms(
            base_link_to_world[:3],
            base_link_to_world[3:],
            world_to_obj[:3],
            world_to_obj[3:],
        )
    )
    base_link_to_held_obj = p.invertTransform(*held_obj_to_base_link)

    # Pick up the cup to demonstrate that we can.
    cup_transform = Pose((-0.1, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0))
    new_cup_pose = multiply_poses(cup_pose, cup_transform)
    fingers_to_cup = multiply_poses(
        cup_pose.invert(),
        get_link_pose(robot.robot_id, finger_frame_id, physics_client_id),
    )
    new_fingers_pose = multiply_poses(new_cup_pose, fingers_to_cup)

    new_collision_ids = set(collision_ids) - {held_obj_id}
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
        img = capture_image(
            physics_client_id,
            camera_yaw=180,
            camera_distance=2.5,
            camera_pitch=-20,
            image_width=900,
        )
        imgs.append(img)

    iio.mimsave("motion_planning_example.mp4", imgs, fps=20)


if __name__ == "__main__":
    _main()
