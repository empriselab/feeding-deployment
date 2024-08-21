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
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.single_arm import SingleArmTwoFingerGripperPyBulletRobot
from pybullet_helpers.utils import create_pybullet_block
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class SceneDescription:
    """Scene description."""

    # Robot.
    robot_name: str = "kinova-gen3"
    initial_joints: JointPositions = field(
        default_factory=lambda: [
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
    )
    robot_base_pose: Pose = Pose(
        (0.0, 0.0, 0.0),
        tuple(Rotation.from_euler("xyz", [0, 0, 90], degrees=True).as_quat()), # Rajat ToDo: Base sure robot-base orientation is 0,0,0
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
    table_pose: Pose = Pose((-0.5, 0.75, -0.29))
    table_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    table_half_extents: tuple[float, float, float] = (0.75, 0.25, 0.5)

    # Cup.
    cup_rgba: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    cup_radius: float = 0.03
    cup_length: float = 0.15
    cup_pose: Pose = Pose((0.0, 0.75, cup_length / 2 + 0.21))
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
            "cup_pose",
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

        # TODO fix...
        physics_client_id = p.connect(p.DIRECT)
        scene = create_pybullet_scene_from_description(physics_client_id, self)
        robot = scene.robot
        joint_infos = get_joint_infos(
            robot.robot_id, robot.arm_joints, robot.physics_client_id
        )
        diff = get_jointwise_difference(
            joint_infos, self.initial_joints, other.initial_joints
        )
        close = np.allclose(diff, 0, atol=atol)
        p.disconnect(physics_client_id)
        if not close:
            return False

        for fld in fields(self):
            if fld.name == "initial_joints":
                continue  # handled above
            mine, theirs = getattr(self, fld.name), getattr(other, fld.name)
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
