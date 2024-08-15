"""Utilities for cup manipulation."""

from pybullet_helpers.inverse_kinematics import (
    set_robot_joints_with_held_object,
)
from pybullet_helpers.geometry import Pose
from pybullet_helpers.joint import (
    JointPositions,
)
from dataclasses import dataclass
from pybullet_helpers.camera import capture_superimposed_image
import imageio.v2 as iio
from pathlib import Path
import pybullet as p

from scene import (
    CupManipulationSceneDescription,
    CupManipulationSceneIDs,
)


@dataclass
class CupManipulationTrajectory:
    """A trajectory for cup manipulation."""

    joint_states: list[JointPositions]
    held_cup_transforms: list[Pose | None]

    def __post_init__(self) -> None:
        assert len(self.joint_states) == len(self.held_cup_transforms)


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
    iio.mimsave(outfile, imgs, fps=fps)
    print(f"Wrote out to {outfile}")
    p.removeBody(cup_target_id, physicsClientId=physics_client_id)
