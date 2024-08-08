"""Demo using https://github.com/tomsilver/pybullet-helpers."""

from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.camera import create_gui_connection
from pybullet_helpers.utils import create_pybullet_block
import numpy as np
import time

import pybullet as p


def _main():
    seed = 0
    collision_region_orientation = (1.0, 0.0, 0.0, 0.0)
    collision_region_position = (0.5, 0.25, 0.25)
    collision_region_half_extents = (0.25, 0.25, 0.5)
    collision_region_good_rgba = (0.0, 1.0, 0.0, 0.5)
    collision_region_bad_rgba = (1.0, 0.0, 0.0, 0.5)

    physics_client_id = create_gui_connection()
    robot = create_pybullet_robot("kinova-gen3", physics_client_id)
    collision_region_id = create_pybullet_block(
        collision_region_good_rgba,
        half_extents=collision_region_half_extents,
        mass=-1,
        friction=0.0,
        orientation=collision_region_orientation,
        physics_client_id=physics_client_id,
    )
    p.resetBasePositionAndOrientation(
        collision_region_id,
        collision_region_position,
        collision_region_orientation,
        physicsClientId=physics_client_id,
    )

    rng = np.random.default_rng(seed)
    while True:
        # Add some random noise to the current joints.
        joint_states = robot.get_joint_positions()
        noise = rng.normal(loc=0.0, scale=0.1, size=len(joint_states))
        joint_states = np.clip(
            joint_states + noise, robot.action_space.low, robot.action_space.high
        )
        # Update the joint states.
        robot.set_joints(joint_states)
        # Check for collisions.
        p.performCollisionDetection(physicsClientId=physics_client_id)
        if p.getContactPoints(
            robot.robot_id, collision_region_id, physicsClientId=physics_client_id
        ):
            color = collision_region_bad_rgba
        else:
            color = collision_region_good_rgba
        p.changeVisualShape(
            collision_region_id, -1, rgbaColor=color, physicsClientId=physics_client_id
        )
        time.sleep(0.01)


if __name__ == "__main__":
    _main()
