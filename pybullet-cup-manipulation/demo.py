"""Script to develop cup manipulation skills in simulation."""

from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.single_arm import (
    SingleArmPyBulletRobot,
    SingleArmTwoFingerGripperPyBulletRobot,
)
from pybullet_helpers.inverse_kinematics import sample_collision_free_inverse_kinematics
from pybullet_helpers.ikfast.utils import get_ikfast_joints
from pybullet_helpers.geometry import Pose, get_pose, multiply_poses
from pybullet_helpers.joint import (
    JointPositions,
    get_jointwise_difference,
    get_joint_info,
)
from pybullet_helpers.camera import capture_image
from pybullet_helpers.utils import create_pybullet_cylinder, create_pybullet_block
from pybullet_helpers.motion_planning import run_motion_planning
from pybullet_helpers.gui import visualize_pose, create_gui_connection
from pybullet_helpers.geometry import matrix_from_quat
from pybullet_helpers.math_utils import get_poses_facing_line
import numpy as np
from pathlib import Path
import imageio.v2 as iio
from tqdm import tqdm

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

    collision_region_id2 = create_pybullet_block(
        collision_region2_rgba,
        half_extents=collision_region2_half_extents,
        physics_client_id=physics_client_id,
    )
    p.resetBasePositionAndOrientation(
        collision_region_id2,
        collision_region2_position,
        collision_region2_orientation,
        physicsClientId=physics_client_id,
    )
    collision_region_ids.add(collision_region_id2)

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


def _score_motion_plan(
    robot: SingleArmPyBulletRobot,
    motion_plan: list[JointPositions],
    joint_geometric_scalar: float = 0.9,
) -> float:
    """Lower is better."""
    # TODO move to pybullet-helpers
    # TODO don't assume ikfast and clean this up...
    joint_infos, _ = get_ikfast_joints(robot)

    if isinstance(robot, SingleArmTwoFingerGripperPyBulletRobot):
        first_finger_idx, second_finger_idx = sorted(
            [robot.left_finger_joint_idx, robot.right_finger_joint_idx]
        )
        first_finger_joint_info = get_joint_info(
            robot.robot_id, first_finger_idx, robot.physics_client_id
        )
        second_finger_joint_info = get_joint_info(
            robot.robot_id, second_finger_idx, robot.physics_client_id
        )
        joint_infos.insert(first_finger_idx, first_finger_joint_info)
        joint_infos.insert(second_finger_idx, second_finger_joint_info)

    score = 0.0

    weights = [1.0]
    num_joints = len(motion_plan[0])
    for i in range(num_joints - 1):
        weights.append(weights[-1] * joint_geometric_scalar)

    for t in range(len(motion_plan) - 1):
        q1, q2 = motion_plan[t], motion_plan[t + 1]
        diff = get_jointwise_difference(joint_infos, q2, q1)
        dist = np.abs(diff)
        score += np.sum(weights * dist)

    return score


def _main():
    seed = 0

    robot, cup_id, table_id, other_collision_ids = _initialize_scene()
    collision_ids = {cup_id, table_id} | other_collision_ids
    physics_client_id = robot.physics_client_id
    robot_initial_joints = robot.get_joint_positions()

    # Use the table pose as a frame of reference.
    table_frame = get_pose(table_id, physics_client_id)
    # Move the frame to the bottom right hand corner of the table so we can see it.
    dims = p.getVisualShapeData(table_id, physicsClientId=physics_client_id)[0][3]
    offset = Pose((dims[0] / 2, -dims[1] / 2, dims[2] / 2))
    table_frame = multiply_poses(offset, table_frame)

    visualize_pose(table_frame, physics_client_id)

    # Find target end effector pose relative to the cup.
    cup_pose = get_pose(cup_id, physics_client_id)

    # Find a number of possible target joint positions.
    max_grasp_candidates = 25
    max_ik_candidates_per_grasp = 100
    rng = np.random.default_rng(seed)
    all_target_joint_positions = []
    for _ in range(max_grasp_candidates):
        candidate_grasp = _sample_grasp(cup_pose, rng)
        for candidate_joints in sample_collision_free_inverse_kinematics(
            robot,
            candidate_grasp,
            collision_ids,
            max_candidates=max_ik_candidates_per_grasp,
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
            physics_client_id,
        )
        if plan is not None:
            all_motion_plans.append(plan)

    print(f"Found {len(all_motion_plans)} motion plans.")

    # Choose the best motion plan.
    plan = min(all_motion_plans, key=lambda v: _score_motion_plan(robot, v))

    imgs = []
    for state in plan:
        robot.set_joints(state)
        img = capture_image(
            physics_client_id, camera_yaw=180, camera_distance=2.5, camera_pitch=-35
        )
        imgs.append(img)

    iio.mimsave("motion_planning_example.mp4", imgs, fps=20)


if __name__ == "__main__":
    _main()
