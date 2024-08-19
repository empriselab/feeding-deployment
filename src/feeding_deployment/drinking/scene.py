"""Create a cup manipulation PyBullet scene given poses for key objects."""

from __future__ import annotations
from dataclasses import dataclass, replace, fields
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.geometry import Pose, Pose3D, Quaternion, multiply_poses
from pybullet_helpers.robots.single_arm import SingleArmTwoFingerGripperPyBulletRobot
from pybullet_helpers.utils import create_pybullet_block
from pybullet_helpers.math_utils import rotate_about_point

from scipy.spatial.transform import Rotation

from pathlib import Path
import pybullet as p
from typing import Any
import numpy as np


@dataclass(frozen=True)
class CupManipulationSceneDescription:
    """Scene description for a cup manipulation scene."""

    # Robot.
    robot_name: str = "kinova-gen3"
    initial_joints: tuple[JointPositions] = (
        np.pi / 2,
        -np.pi / 4,
        -np.pi / 2,
        0.0,
        np.pi / 2,
        -np.pi / 2,
        np.pi / 2,
        0.0,
        0.0,
    )
    robot_base_pose: Pose = Pose(
        (0.0, 0.0, 0.0),
        tuple(Rotation.from_euler("xyz", [0, 0, 90], degrees=True).as_quat()),
    )

    # Robot holder (vention stand).
    robot_holder_pose: Pose = Pose((0.0, 0.0, -0.5 - 0.05))
    robot_holder_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    robot_holder_half_extents: tuple[float, float, float] = (0.25, 0.25, 0.5)

    # Wheelchair.
    wheelchair_pose: Pose = Pose((-0.5, 0.0, -0.25), (0.0, 0.0, 1.0, 0.0))
    wheelchair_relative_head_pose: Pose = Pose((0.0, -0.25, 0.75), (0.0, 0.0, 0.0, 1.0))
    wheelchair_urdf_path: Path = (
        Path(__file__).parent.parent
        / "assets"
        / "urdf"
        / "wheelchair"
        / "wheelchair.urdf"
    )

    # Table.
    table_pose: Pose = Pose((-0.5, 0.75, -0.5))
    table_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    table_half_extents: tuple[float, float, float] = (0.75, 0.25, 0.5)

    # Cup.
    cup_rgba: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    cup_radius: float = 0.03
    cup_length: float = 0.15
    cup_pose: Pose = Pose((0.0, 0.75, cup_length / 2))
    cup_handle_half_extents: tuple[float, float, float] = (
        cup_radius,
        cup_radius,
        cup_radius,
    )
    cup_handle_rgba: tuple[float, float, float, float] = (0.7, 0.2, 0.5, 1.0)
    cup_handle_relative_pose: Pose = Pose(
        (
            0.0,
            -cup_radius - cup_handle_half_extents[1] / 2,
            cup_length / 4,
        )
    )
    cup_grasp_distance: float = 0.075
    cup_grasp_transform: Pose = Pose(
        (0.0, -cup_grasp_distance, 0.0),
        p.getQuaternionFromEuler((np.pi / 2, np.pi, np.pi)),
    )

    # Staging pose (where the drinking motion planning should finish).
    # This is relative to the wheelchair head.
    staging_relative_pose: Pose = Pose(
        (-0.1, 0.5, 0.0), p.getQuaternionFromEuler((0.0, 0.0, np.pi / 2))
    )

    @property
    def wheelchair_head_pose(self) -> Pose:
        """Derived from wheelchair base and relative pose."""
        # The wheelchair is weirdly flipped in the URDF, so correct for that.
        flip_tf = Pose(
            (0.0, 0.0, 0.0), tuple(p.getQuaternionFromEuler((0.0, 0.0, np.pi)))
        )
        flipped_wheelchair_pose = multiply_poses(self.wheelchair_pose, flip_tf)
        wheelchair_center_pose = Pose(
            flipped_wheelchair_pose.position, flipped_wheelchair_pose.orientation
        )
        return multiply_poses(
            wheelchair_center_pose, self.wheelchair_relative_head_pose
        )

    @property
    def cup_staging_pose(self) -> Pose:
        """Target pose for the cup before transfer."""
        return multiply_poses(self.wheelchair_head_pose, self.staging_relative_pose)

    @property
    def camera_kwargs(self) -> dict[str, Any]:
        """Derived kwargs for taking images."""
        base_position = self.robot_base_pose.position
        head_position = self.wheelchair_head_pose.position
        return dict(
            outer_camera_target=base_position,
            outer_camera_yaw=180,
            outer_camera_distance=2.5,
            inner_camera_target=head_position,
            inner_camera_yaw=0,
            inner_camera_distance=1.0,
            inner_camera_pitch=-20,
        )

    def rotate_about_point(
        self, point: Pose3D, rotation: Quaternion
    ) -> CupManipulationSceneDescription:
        """Create a rotated scene (useful for testing.)"""
        pose_fields = {
            "robot_base_pose",
            "robot_holder_pose",
            "wheelchair_pose",
            "table_pose",
            "cup_pose",
        }
        pose_dict = {
            k: rotate_about_point(point, rotation, getattr(self, k))
            for k in pose_fields
        }
        return replace(
            self,
            **pose_dict,
        )

    def allclose(self, other: Any, atol=1e-5) -> bool:
        if not isinstance(other, CupManipulationSceneDescription):
            return False
        for field in fields(self):
            mine, theirs = getattr(self, field.name), getattr(other, field.name)
            if hasattr(mine, "allclose"):
                field_close = mine.allclose(theirs, atol=atol)
            elif isinstance(mine, (tuple, list)):
                field_close = np.allclose(mine, theirs, atol=atol)
            elif isinstance(mine, (float, int)):
                field_close = np.isclose(mine, theirs, atol=atol)
            else:
                field_close = mine == theirs
            if not field_close:
                return False
        return True


@dataclass(frozen=True)
class CupManipulationSceneIDs:
    """Holds the PyBullet IDs for objects in a cup manipulation scene."""

    physics_client_id: int
    robot: SingleArmTwoFingerGripperPyBulletRobot
    robot_holder_id: int
    wheelchair_id: int
    table_id: int
    cup_id: int
    cup_handle_link_id: int

    def reset(self, scene_description: CupManipulationSceneDescription):
        """Reset the scene from a description."""
        # Reset the robot.
        self.robot.set_joints(scene_description.initial_joints)

        # Reset the cup.
        p.resetBasePositionAndOrientation(
            self.cup_id,
            scene_description.cup_pose.position,
            scene_description.cup_pose.orientation,
            physicsClientId=self.physics_client_id,
        )


def create_cup_manipulation_scene(
    physics_client_id: int, scene_description: CupManipulationSceneDescription
) -> CupManipulationSceneIDs:
    """Build a cup manipulation scene in PyBullet from a description."""

    # Create robot.
    robot = create_pybullet_robot(
        scene_description.robot_name,
        physics_client_id,
        base_pose=scene_description.robot_base_pose,
        control_mode="reset",
        home_joint_positions=scene_description.initial_joints,
    )

    # Create a holder.
    robot_holder_id = create_pybullet_block(
        scene_description.robot_holder_rgba,
        half_extents=scene_description.robot_holder_half_extents,
        physics_client_id=physics_client_id,
    )
    p.resetBasePositionAndOrientation(
        robot_holder_id,
        scene_description.robot_holder_pose.position,
        scene_description.robot_holder_pose.orientation,
        physicsClientId=physics_client_id,
    )

    # Create wheelchair.
    wheelchair_id = p.loadURDF(
        str(scene_description.wheelchair_urdf_path),
        useFixedBase=True,
        physicsClientId=physics_client_id,
    )
    p.resetBasePositionAndOrientation(
        wheelchair_id,
        scene_description.wheelchair_pose.position,
        scene_description.wheelchair_pose.orientation,
        physicsClientId=physics_client_id,
    )

    # Create cup.
    cup_collision_id = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=scene_description.cup_radius,
        height=scene_description.cup_length,
        physicsClientId=physics_client_id,
    )
    cup_visual_id = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=scene_description.cup_radius,
        length=scene_description.cup_length,
        rgbaColor=scene_description.cup_rgba,
        physicsClientId=physics_client_id,
    )
    cup_handle_collision_id = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=scene_description.cup_handle_half_extents,
        physicsClientId=physics_client_id,
    )
    cup_handle_visual_id = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=scene_description.cup_handle_half_extents,
        rgbaColor=scene_description.cup_handle_rgba,
        physicsClientId=physics_client_id,
    )
    cup_id = p.createMultiBody(
        baseMass=-1,
        baseCollisionShapeIndex=cup_collision_id,
        baseVisualShapeIndex=cup_visual_id,
        basePosition=scene_description.cup_pose.position,
        baseOrientation=scene_description.cup_pose.orientation,
        linkMasses=[-1],
        linkCollisionShapeIndices=[cup_handle_collision_id],
        linkVisualShapeIndices=[cup_handle_visual_id],
        linkPositions=[scene_description.cup_handle_relative_pose.position],
        linkOrientations=[scene_description.cup_handle_relative_pose.orientation],
        linkInertialFramePositions=[(0.0, 0.0, 0.0)],
        linkInertialFrameOrientations=[(0.0, 0.0, 0.0, 1.0)],
        linkParentIndices=[0],
        linkJointTypes=[p.JOINT_FIXED],
        linkJointAxis=[(0.0, 0.0, 1.0)],
        physicsClientId=physics_client_id,
    )
    cup_handle_link_id = 0

    # Create a table.
    table_id = create_pybullet_block(
        scene_description.table_rgba,
        half_extents=scene_description.table_half_extents,
        physics_client_id=physics_client_id,
    )
    p.resetBasePositionAndOrientation(
        table_id,
        scene_description.table_pose.position,
        scene_description.table_pose.orientation,
        physicsClientId=physics_client_id,
    )

    return CupManipulationSceneIDs(
        physics_client_id,
        robot,
        robot_holder_id,
        wheelchair_id,
        table_id,
        cup_id,
        cup_handle_link_id,
    )
