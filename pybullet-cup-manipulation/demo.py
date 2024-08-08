"""Script to develop cup manipulation skills in simulation."""

from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot
from pybullet_helpers.geometry import Pose
from pybullet_helpers.camera import create_gui_connection
from pybullet_helpers.utils import create_pybullet_cylinder, create_pybullet_block
import numpy as np
import time
from pathlib import Path

import pybullet as p


def _initialize_scene() -> tuple[SingleArmPyBulletRobot, int, int, int]:
    """Returns robot, cup ID, table ID, collision ID."""

    robot_base_pose = Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    robot_home_joints = [
        -np.pi / 2,
        -np.pi / 4,
        -np.pi / 2,
        0.,
        np.pi / 2,
        -np.pi / 2,
        np.pi / 2,
        0.,
        0.,
    ]

    robot_holder_rgba = (0.5, 0.5, 0.5, 1.0)
    robot_holder_half_extents = (0.25, 0.25, 0.5)
    robot_holder_position = (0.0, 0.0, -0.5)
    robot_holder_orientation = (1.0, 0.0, 0.0, 0.0)

    collision_region_orientation = (1.0, 0.0, 0.0, 0.0)
    collision_region_position = (0.75, 0.3, 0.0)
    collision_region_half_extents = (0.5, 0.75, 1.0)
    collision_region_rgba = (1.0, 0.0, 0.0, 0.0)

    wheelchair_position = (0.5, 0.0, -0.25)
    wheelchair_orientation = (0.0, 0.0, 0.0, 1.0)

    table_rgba = (0.5, 0.5, 0.5, 1.0)
    table_half_extents = (0.75, 0.25, 0.5)
    table_position = (0.5, -0.75, -0.5)
    table_orientation = (1.0, 0.0, 0.0, 0.0)

    cup_rgba = (0.0, 0.0, 1.0, 1.0)
    cup_radius = 0.04
    cup_length = 0.12
    cup_position = (0.0, -0.75, cup_length / 2)
    cup_orientation = (1.0, 0.0, 0.0, 0.0)

    physics_client_id = create_gui_connection()

    # Create robot.
    robot = create_pybullet_robot("kinova-gen3", physics_client_id,
                                  base_pose=robot_base_pose,
                                  home_joint_positions=robot_home_joints)
    
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

    # Create collision area.
    collision_region_id = create_pybullet_block(
        collision_region_rgba,
        half_extents=collision_region_half_extents,
        physics_client_id=physics_client_id,
    )
    p.resetBasePositionAndOrientation(
        collision_region_id,
        collision_region_position,
        collision_region_orientation,
        physicsClientId=physics_client_id,
    )

    return robot, cup_id, table_id, collision_region_id


def _main():

    robot, cup_id, table_id, collision_region_id = _initialize_scene()
    physics_client_id = robot.physics_client_id

    while True:
        p.stepSimulation(physicsClientId=physics_client_id)
        time.sleep(0.01)


if __name__ == "__main__":
    _main()
