"""Utilities for cup manipulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import imageio.v2 as iio
import numpy as np
import pybullet as p
from numpy.typing import NDArray
from pybullet_helpers.camera import capture_superimposed_image
from pybullet_helpers.geometry import Pose
from pybullet_helpers.inverse_kinematics import (
    set_robot_joints_with_held_object,
)
from pybullet_helpers.joint import (
    JointPositions,
)

from feeding_deployment.drinking.cup_manipulation_scene import (
    CupManipulationSceneDescription,
    CupManipulationSceneIDs,
)


@dataclass
class CupManipulationTrajectory:
    """A trajectory for cup manipulation."""

    joint_states: list[JointPositions] = field(default_factory=lambda: [])
    held_cup_transforms: list[Pose | None] = field(default_factory=lambda: [])

    def __post_init__(self) -> None:
        assert len(self.joint_states) == len(self.held_cup_transforms)

    def extend(self, other: CupManipulationTrajectory) -> None:
        """Extend this trajectory in-place."""
        self.joint_states.extend(other.joint_states)
        self.held_cup_transforms.extend(other.held_cup_transforms)


def get_kinova_controller_trajectory(
    traj: CupManipulationTrajectory,
) -> list[tuple[NDArray, float]]:
    """The Kinova controller expects arm joints and gripper values."""
    cmds = []
    for joint_state in traj.joint_states:
        assert len(joint_state) == 9  # making assumptions about Kinova
        arm = np.array(joint_state[:7])
        assert np.isclose(joint_state[7], joint_state[8])
        gripper = joint_state[8]
        cmds.append((arm, gripper))
    return cmds


def make_cup_manipulation_video(
    scene: CupManipulationSceneIDs,
    scene_description: CupManipulationSceneDescription,
    traj: CupManipulationTrajectory,
    outfile: Path,
    fps: int = 20,
    cup_target_opacity: float = 0.75,
) -> None:
    """Make a video for a simulated cup manipulation plan."""
    scene.reset(scene_description)
    physics_client_id = scene.physics_client_id
    imgs = []
    cup_target_id = None
    for joint_state, base_link_to_held_obj in zip(
        traj.joint_states, traj.held_cup_transforms, strict=True
    ):
        if base_link_to_held_obj is None:
            scene.robot.set_joints(joint_state)
        else:

            if cup_target_id is None:
                # Create visual shape only to show target.
                cup_target_visual_id = p.createVisualShape(
                    p.GEOM_CYLINDER,
                    radius=scene_description.cup_radius,
                    length=scene_description.cup_length,
                    rgbaColor=scene_description.cup_rgba[:3] + (cup_target_opacity,),
                    physicsClientId=physics_client_id,
                )
                cup_handle_target_visual_id = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=scene_description.cup_handle_half_extents,
                    rgbaColor=scene_description.cup_handle_rgba[:3]
                    + (cup_target_opacity,),
                    physicsClientId=physics_client_id,
                )
                cup_target_id = p.createMultiBody(
                    baseMass=-1,
                    baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=cup_target_visual_id,
                    basePosition=scene_description.cup_staging_pose.position,
                    baseOrientation=scene_description.cup_staging_pose.orientation,
                    linkMasses=[-1],
                    linkCollisionShapeIndices=[-1],
                    linkVisualShapeIndices=[cup_handle_target_visual_id],
                    linkPositions=[scene_description.cup_handle_relative_pose.position],
                    linkOrientations=[
                        scene_description.cup_handle_relative_pose.orientation
                    ],
                    linkInertialFramePositions=[(0.0, 0.0, 0.0)],
                    linkInertialFrameOrientations=[(0.0, 0.0, 0.0, 1.0)],
                    linkParentIndices=[0],
                    linkJointTypes=[p.JOINT_FIXED],
                    linkJointAxis=[(0.0, 0.0, 1.0)],
                    physicsClientId=physics_client_id,
                )

            set_robot_joints_with_held_object(
                scene.robot,
                scene.physics_client_id,
                scene.cup_id,
                base_link_to_held_obj,
                joint_state,
            )
        img = capture_superimposed_image(
            scene.physics_client_id, **scene_description.camera_kwargs
        )
        imgs.append(img)
    iio.mimsave(outfile, imgs, fps=fps)  # type: ignore
    print(f"Wrote out to {outfile}")
    if cup_target_id is not None:
        p.removeBody(cup_target_id, physicsClientId=physics_client_id)
