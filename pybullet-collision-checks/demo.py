"""Demonstrate collision checking with Kinova Gen3."""

import pybullet as p
from typing import TypeAlias
from pathlib import Path
import numpy as np


Pose3D: TypeAlias = tuple[float, float, float]


def create_pybullet_gui_connection(
        camera_distance: float = 1.5,
        camera_yaw: float = 0,
        camera_pitch: float = -24,
        camera_target: Pose3D = (0.0, 0.0, 0.5),
        disable_preview_windows: bool = True) -> int:
    """Creates a PyBullet GUI connection and initializes the camera.

    Returns the physics client ID for the connection.

    Not covered by unit tests because unit tests need to be headless.
    """
    physics_client_id = p.connect(p.GUI, options=('--background_color_red=0.0 '
                                                  '--background_color_green=0.0 '
                                                  '--background_color_blue=0.0'))
    # Disable the PyBullet GUI preview windows for faster rendering.
    if disable_preview_windows:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,
                                   False,
                                   physicsClientId=physics_client_id)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW,
                                   False,
                                   physicsClientId=physics_client_id)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                                   False,
                                   physicsClientId=physics_client_id)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
                                   False,
                                   physicsClientId=physics_client_id)
    p.resetDebugVisualizerCamera(camera_distance,
                                 camera_yaw,
                                 camera_pitch,
                                 camera_target,
                                 physicsClientId=physics_client_id)
    return physics_client_id


def get_asset_path(asset_name: str) -> Path:
    """Return the absolute path to env asset."""
    dir_path = Path(__file__).parent
    path = dir_path / "assets" / asset_name
    assert path.exists(), f"Asset not found: {asset_name}."
    return path


def _main():
    seed = 0
    num_steps = 100
    robot_base_position = (0, 0, 0)
    collision_region_orientation = (1, 0, 0, 0)
    collision_region_position = (0.5, 0.25, 0.25)
    collision_region_half_extents = (0.25, 0.25, 0.5)
    collision_region_good_rgba = (0.0, 1.0, 0.0, 0.5)
    collision_region_bad_rgba = (1.0, 0.0, 0.0, 0.5)

    physics_client_id = create_pybullet_gui_connection()

    p.resetSimulation(physicsClientId=physics_client_id)
    
    # Load the Kinova Gen3 robot.
    robot_id =p.loadURDF(str(get_asset_path("urdf/kinova_gen3/GEN3_URDF_V12.urdf")),
               basePosition=robot_base_position,
               useFixedBase=True,
               physicsClientId=physics_client_id)
    
    # Create a region next to the robot where we want to avoid collisions.
    # Create the collision shape.
    collision_id = p.createCollisionShape(p.GEOM_BOX,
                                          halfExtents=collision_region_half_extents,
                                          physicsClientId=physics_client_id)

    # Create the visual_shape.
    visual_id = p.createVisualShape(p.GEOM_BOX,
                                    halfExtents=collision_region_half_extents,
                                    rgbaColor=collision_region_good_rgba,
                                    physicsClientId=physics_client_id)

    # Create the body.
    block_id = p.createMultiBody(baseMass=-1,
                                 baseCollisionShapeIndex=collision_id,
                                 baseVisualShapeIndex=visual_id,
                                 basePosition=collision_region_position,
                                 baseOrientation=collision_region_orientation,
                                 physicsClientId=physics_client_id)
    
    # Do some random joint movements and visualize when there is a collision.
    joint_idxs = list(range(p.getNumJoints(robot_id, physicsClientId=physics_client_id)))
    joint_infos = [p.getJointInfo(robot_id, idx, physicsClientId=physics_client_id)
                   for idx in joint_idxs]
    joint_limits = [(j[8], j[9]) for j in joint_infos]
    joint_states = [p.getJointState(robot_id, i, physicsClientId=physics_client_id)[0]
                    for i in joint_idxs]
    
    rng = np.random.default_rng(seed)
    for _ in range(num_steps):
        # Add some random noise to the current joints.
        new_joint_states = np.zeros_like(joint_states)
        for i, v in enumerate(joint_states):
            l, h = joint_limits[i]
            new_joint_states[i] = np.clip(v + rng.normal(0.0, scale=0.1), l, h)
        joint_states = new_joint_states
        # Update the joint states.
        for i, v in enumerate(joint_states):
            p.resetJointState(robot_id, i, v, targetVelocity=0.0,
                              physicsClientId=physics_client_id)
        # Check for collisions.
        p.performCollisionDetection(physicsClientId=physics_client_id)
        if p.getContactPoints(robot_id,
                              block_id,
                              physicsClientId=physics_client_id):
            color = collision_region_bad_rgba
        else:
            color = collision_region_good_rgba
        p.changeVisualShape(block_id, -1, rgbaColor=color,
                            physicsClientId=physics_client_id)

        import time; time.sleep(0.1)


if __name__ == "__main__":
    _main()
