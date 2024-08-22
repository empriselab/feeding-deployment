"""The description of a simulation initial state, with default values."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as p
from pybullet_helpers.geometry import Pose, Pose3D, Quaternion, multiply_poses
from pybullet_helpers.joint import (
    JointPositions,
    get_joint_infos,
    get_jointwise_difference,
)
from pybullet_helpers.math_utils import rotate_about_point
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class SceneDescription:
    """Scene description."""

    # Robot.
    robot_name: str = "kinova-gen3"
    initial_joints: JointPositions = field(
        default_factory=lambda: [
            0,
            -np.pi / 4,
            -np.pi / 2,
            0.0,
            np.pi / 2,
            -np.pi / 2,
            np.pi / 2,
            0.0,
            0.0,
        ]
    )
    robot_base_pose: Pose = Pose(
        (0.0, 0.0, 0.0),
        tuple(
            Rotation.from_euler("xyz", [0, 0, 90], degrees=True).as_quat()
        ),  # Rajat ToDo: Base sure robot-base orientation is 0,0,0
    )

    # Robot holder (vention stand).
    robot_holder_pose: Pose = Pose((0.0, 0.0, -0.55))
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

    # Conservative bounding box around the wheel chair.
    conservative_bb_pose: Pose = Pose((-0.75, -0.5, -0.25))
    conservative_bb_rgba: tuple[float, float, float, float] = (0.9, 0.1, 0.1, 0.5)
    conservative_bb_half_extents: tuple[float, float, float] = (0.4, 0.4, 1.0)

    # Table.
    table_pose: Pose = Pose((-0.5, 0.75, -0.29))
    table_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    table_half_extents: tuple[float, float, float] = (0.75, 0.25, 0.5)

    # Cup.
    cup_pose: Pose = Pose(
        (0.0, 0.75, 0.35), p.getQuaternionFromEuler((np.pi / 2, 0.0, np.pi))
    )
    cup_urdf_path: Path = (
        Path(__file__).parent.parent
        / "assets"
        / "urdf"
        / "drinking_utensil"
        / "drinking_utensil.urdf"
    )

    cup_grasp_fingers_orientation: Quaternion = p.getQuaternionFromEuler(
        (np.pi, np.pi, 0)
    )
    cup_pregrasp_transform: Pose = Pose(
        (0.0, 0.0, -0.1),
        cup_grasp_fingers_orientation,
    )
    cup_grasp_transform: Pose = Pose(
        (0.0, 0.0, -0.025),
        cup_grasp_fingers_orientation,
    )
    cup_prestow_transform: Pose = Pose(
        (0.0, 0.05, -0.025),
        cup_grasp_fingers_orientation,
    )
    # This is relative to the wheelchair head.
    cup_staging_transform: Pose = Pose(
        (-0.1, 0.5, 0.0), p.getQuaternionFromEuler((-np.pi / 2, np.pi, np.pi / 2))
    )
    cup_transfer_transform: Pose = Pose(
        (-0.1, 0.35, 0.0),
        p.getQuaternionFromEuler((-np.pi / 2, np.pi - np.pi / 8, np.pi / 2)),
    )

    # Wiper.
    wiper_pose: Pose = Pose(
        (0.35, 0.15, -0.05), p.getQuaternionFromEuler((0.0, np.pi, np.pi))
    )
    wiper_urdf_path: Path = (
        Path(__file__).parent.parent
        / "assets"
        / "urdf"
        / "wiping_utensil"
        / "wiping_utensil.urdf"
    )
    wiper_grasp_fingers_orientation: Quaternion = p.getQuaternionFromEuler((0, 0, 0))
    wiper_pregrasp_transform: Pose = Pose(
        (0.0, 0.0, -0.1),
        wiper_grasp_fingers_orientation,
    )
    wiper_grasp_transform: Pose = Pose(
        (0.0, 0.0, -0.025),
        wiper_grasp_fingers_orientation,
    )
    wiper_prestow_transform: Pose = Pose(
        (0.0, 0.0, -0.2),
        wiper_grasp_fingers_orientation,
    )
    # This is relative to the wheelchair head.
    wiper_staging_transform: Pose = Pose(
        (-0.1, 0.5, 0.0), p.getQuaternionFromEuler((-np.pi / 2, np.pi, np.pi / 2))
    )

    # Utensil.
    utensil_pose: Pose = Pose(
        (0.35, -0.15, -0.05), p.getQuaternionFromEuler((0.0, np.pi, np.pi))
    )
    utensil_urdf_path: Path = (
        Path(__file__).parent.parent
        / "assets"
        / "urdf"
        / "feeding_utensil"
        / "feeding_utensil.urdf"
    )
    utensil_grasp_fingers_orientation: Quaternion = p.getQuaternionFromEuler((0, 0, 0))
    utensil_pregrasp_transform: Pose = Pose(
        (0.0, 0.0, -0.1),
        utensil_grasp_fingers_orientation,
    )
    utensil_grasp_transform: Pose = Pose(
        (0.0, 0.0, -0.025),
        wiper_grasp_fingers_orientation,
    )
    utensil_prestow_transform: Pose = Pose(
        (0.0, 0.0, -0.2),
        wiper_grasp_fingers_orientation,
    )
    # This is relative to the wheelchair head.
    utensil_staging_transform: Pose = Pose(
        (-0.1, 0.5, 0.0), p.getQuaternionFromEuler((-np.pi / 2, np.pi, np.pi / 2))
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
    def cup_pregrasp_pose(self) -> Pose:
        """Pose for the finger tip to pregrasp the cup."""
        return multiply_poses(self.cup_pose, self.cup_pregrasp_transform)

    @property
    def cup_grasp_pose(self) -> Pose:
        """Pose for the finger tip to grasp the cup."""
        return multiply_poses(self.cup_pose, self.cup_grasp_transform)

    @property
    def cup_prestow_pose(self) -> Pose:
        """Pose for the finger tip to prestow the cup."""
        return multiply_poses(
            self.cup_pose,
            self.cup_prestow_transform,
        )

    @property
    def cup_staging_pose(self) -> Pose:
        """Pose for the finger tip before cup transfer."""
        target_cup_pose = multiply_poses(
            self.wheelchair_head_pose, self.cup_staging_transform
        )
        fingers_to_cup = self.cup_grasp_transform
        return multiply_poses(target_cup_pose, fingers_to_cup)

    @property
    def cup_transfer_pose(self) -> Pose:
        """Pose for the finger tip for cup transfer."""
        target_cup_pose = multiply_poses(
            self.wheelchair_head_pose,
            self.cup_transfer_transform,
        )
        fingers_to_cup = self.cup_grasp_transform
        return multiply_poses(target_cup_pose, fingers_to_cup)

    @property
    def wiper_pregrasp_pose(self) -> Pose:
        """Pose for the finger tip to pregrasp the wiper."""
        return multiply_poses(self.wiper_pose, self.wiper_pregrasp_transform)

    @property
    def wiper_grasp_pose(self) -> Pose:
        """Pose for the finger tip to grasp the wiper."""
        return multiply_poses(self.wiper_pose, self.wiper_grasp_transform)

    @property
    def wiper_prestow_pose(self) -> Pose:
        """Pose for the finger tip to prestow the wiper."""
        return multiply_poses(
            self.wiper_pose,
            self.wiper_prestow_transform,
        )

    @property
    def wiper_staging_pose(self) -> Pose:
        """Pose for the finger tip before wiper transfer."""
        target_wiper_pose = multiply_poses(
            self.wheelchair_head_pose, self.wiper_staging_transform
        )
        fingers_to_wiper = self.wiper_grasp_transform
        return multiply_poses(target_wiper_pose, fingers_to_wiper)

    @property
    def utensil_pregrasp_pose(self) -> Pose:
        """Pose for the finger tip to pregrasp the utensil."""
        return multiply_poses(self.utensil_pose, self.utensil_pregrasp_transform)

    @property
    def utensil_grasp_pose(self) -> Pose:
        """Pose for the finger tip to grasp the utensil."""
        return multiply_poses(self.utensil_pose, self.utensil_grasp_transform)

    @property
    def utensil_prestow_pose(self) -> Pose:
        """Pose for the finger tip to prestow the utensil."""
        return multiply_poses(
            self.utensil_pose,
            self.utensil_prestow_transform,
        )

    @property
    def utensil_staging_pose(self) -> Pose:
        """Pose for the finger tip before utensil transfer."""
        target_utensil_pose = multiply_poses(
            self.wheelchair_head_pose, self.utensil_staging_transform
        )
        fingers_to_utensil = self.utensil_grasp_transform
        return multiply_poses(target_utensil_pose, fingers_to_utensil)

    @property
    def camera_kwargs(self) -> dict[str, Any]:
        """Derived kwargs for taking images."""
        base_position = self.robot_base_pose.position
        head_position = self.wheelchair_head_pose.position
        return {
            "outer_camera_target": base_position,
            "outer_camera_yaw": 180,
            "outer_camera_distance": 2.5,
            "inner_camera_target": head_position,
            "inner_camera_yaw": 0,
            "inner_camera_distance": 1.0,
            "inner_camera_pitch": -20,
        }

    def rotate_about_point(
        self, point: Pose3D, rotation: Quaternion
    ) -> SceneDescription:
        """Create a rotated scene (useful for testing.)"""
        pose_fields = {
            "robot_base_pose",
            "robot_holder_pose",
            "wheelchair_pose",
            "table_pose",
            "conservative_bb_pose",
            "cup_pose",
            "wiper_pose",
            "utensil_pose",
        }
        pose_dict: dict[str, Any] = {
            k: rotate_about_point(point, rotation, getattr(self, k))
            for k in pose_fields
        }
        return replace(
            self,
            **pose_dict,
        )

    def allclose(self, other: Any, atol=1e-4) -> bool:
        """Compare this scene description to another, e.g., for caching."""
        if not isinstance(other, SceneDescription):
            return False

        # TODO need to refactor to avoid calling a function that no longer
        # exists (create_pybullet_scene_from_description).
        return False

        # physics_client_id = p.connect(p.DIRECT)
        # scene = create_pybullet_scene_from_description(physics_client_id, self)
        # robot = scene.robot
        # joint_infos = get_joint_infos(
        #     robot.robot_id, robot.arm_joints, robot.physics_client_id
        # )
        # diff = get_jointwise_difference(
        #     joint_infos, self.initial_joints, other.initial_joints
        # )
        # close = np.allclose(diff, 0, atol=atol)
        # p.disconnect(physics_client_id)
        # if not close:
        #     return False

        # for fld in fields(self):
        #     if fld.name == "initial_joints":
        #         continue  # handled above
        #     mine, theirs = getattr(self, fld.name), getattr(other, fld.name)
        #     if hasattr(mine, "allclose"):
        #         field_close = mine.allclose(theirs, atol=atol)
        #     elif isinstance(mine, (tuple, list)):
        #         field_close = np.allclose(mine, theirs, atol=atol)
        #     elif isinstance(mine, (float, int)):
        #         field_close = np.isclose(mine, theirs, atol=atol)
        #     else:
        #         field_close = mine == theirs
        #     if not field_close:
        #         return False
        # return True
