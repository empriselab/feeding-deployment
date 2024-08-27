"""The description of a simulation initial state, with default values."""

from __future__ import annotations

import json
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


def create_scene_description_from_config(config_file_path: str) -> SceneDescription:
    # Read the config file
    with open(config_file_path, "r") as file:
        config = json.load(file)

    # convert lists containting 6 elements to Pose objects
    for key in config:
        if isinstance(config[key], list):
            config[key] = Pose(config[key][:3], config[key][3:])

    # Create an instance of SceneDescription using the config
    return SceneDescription(**config)


@dataclass(frozen=True)
class SceneDescription:
    """Scene description."""

    # robot base frame
    utensil_inside_mount: Pose = Pose(
        (-0.147, -0.17, 0.07),
        (0.7071068, -0.7071068, 0.0, 0.0),
    )
    utensil_inside_mount_pos: JointPositions = field(
        default_factory=lambda: [
            2.854928662273119,
            0.5484296235490069,
            2.3664516551307853,
            -2.5354838210986594,
            1.1181978253322737,
            -0.4196300319060411,
            -0.4776571162655596,
        ]
    )

    utensil_outside_mount: Pose = Pose(
        (-0.147, -0.29, 0.07),
        (0.7071068, -0.7071068, 0.0, 0.0),
    )
    utensil_outside_mount_pos: JointPositions = field(
        default_factory=lambda: [
            2.6263072016502855,
            0.6990732614011294,
            2.3072804767669686,
            -2.22978328799399,
            0.9554788158864868,
            -0.6272143841300117,
            -0.4820587889152428,
        ]
    )

    utensil_above_mount: Pose = Pose(
        (-0.147, -0.17, 0.15),
        (0.7071068, -0.7071068, 0.0, 0.0),
    )
    utensil_above_mount_pos: JointPositions = field(
        default_factory=lambda: [
            -2.9830267107059303,
            0.39129809251777437,
            1.8614008644185065,
            -2.4207417918487044,
            0.614389066373381,
            -0.6996630184245269,
            -0.006409696111602692,
        ]
    )

    utensil_infront_mount: Pose = Pose(
        (0.0, -0.17, 0.15),
        (0.7071068, -0.7071068, 0.0, 0.0),
    )
    utensil_infront_mount_pos: JointPositions = field(
        default_factory=lambda: [
            2.835106221647441,
            0.18716812654374576,
            1.7554270267415284,
            -2.5582927305707517,
            0.3492644556371586,
            -0.5794207625752312,
            -0.3984099643402903,
        ]
    )

    utensil_neutral_pos: JointPositions = field(
        default_factory=lambda: [
            2.2912525080624357,
            0.730991513381838,
            2.0830126187361424,
            -2.1737367965371632,
            0.28532185799581516,
            -0.4648462461578422,
            -0.29495787389950756,
        ]
    )

    above_plate_pos: JointPositions = field(
        default_factory=lambda: [
            -2.86495014,
            -1.61460533,
            -2.6115943,
            -1.37673391,
            1.11842806,
            -1.17904586,
            -2.6957422,
        ]
    )

    before_transfer_pos: JointPositions = field(
        default_factory=lambda: [
            -2.86554642,
            -1.61951779,
            -2.60986085,
            -1.37302839,
            1.11779249,
            -1.18028264,
            2.05515862,
        ]
    )

    retract_pos: JointPositions = field(
        default_factory=lambda: [
            0.0,
            -0.34903602299465675,
            -3.141591055693139,
            -2.5482592711638783,
            0.0,
            -0.872688061814757,
            1.57075917569769,
        ]
    )

    # Robot.
    robot_name: str = "kinova-gen3"
    initial_joints: JointPositions = field(
        default_factory=lambda: [
            0.0,
            -0.34903602299465675,
            -3.141591055693139,
            -2.5482592711638783,
            0.0,
            -0.872688061814757,
            1.57075917569769,
            1.0,
            1.0,
        ]
    )
    robot_base_pose: Pose = Pose(
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )

    # Robot holder (vention stand).
    # robot_holder_pose: Pose = Pose((0.0, 0.0, -0.261))
    robot_holder_pose: Pose = Pose((0.0, 0.0, -0.28))
    robot_holder_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    robot_holder_half_extents: tuple[float, float, float] = (0.18, 0.12, 0.26)

    # Wheelchair.
    wheelchair_pose: Pose = Pose(
        (0.2, 0.5, -0.25), tuple(p.getQuaternionFromEuler((0.0, 0.0, np.pi / 2)))
    )
    wheelchair_relative_head_pose: Pose = Pose(
        (0.0, -0.25, 0.75), (0.0, 0.0, 0.0, 1.0)
    )  # Rajat ToDo: Fix this
    wheelchair_urdf_path: Path = (
        Path(__file__).parent.parent
        / "assets"
        / "urdf"
        / "wheelchair"
        / "wheelchair.urdf"
    )

    # Conservative bounding box around the wheel chair.
    conservative_bb_pose: Pose = Pose((-0.4, 0.7, -0.25))
    conservative_bb_rgba: tuple[float, float, float, float] = (0.9, 0.1, 0.1, 0.5)
    conservative_bb_half_extents: tuple[float, float, float] = (0.4, 0.4, 1.0)

    # Table.
    table_pose: Pose = Pose((0.5, 0.75, -0.175))
    table_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    table_half_extents: tuple[float, float, float] = (0.25, 0.4, 0.345)

    # Cup.
    cup_pose: Pose = Pose(
        (0.53, 0.66, 0.32), p.getQuaternionFromEuler((np.pi / 2, 0.0, np.pi))
    )
    cup_urdf_path: Path = (
        Path(__file__).parent.parent
        / "assets"
        / "urdf"
        / "drinking_utensil"
        / "drinking_utensil.urdf"
    )

    # outside_cup_pose = (
    #         np.array([0.54, 0.5, 0.3158]),
    #         np.array([0, 0.7071068, 0.7071068, 0]),
    # )
    # outside_cup_pos = [-2.9471131844578844, -1.6206821277020431, -1.724364315261008, -1.3886978935707663, 0.38836691755349223, -0.42193579677783166, 2.813355918708655]

    # cup_inside_mount = (
    #     np.array([0.54, 0.58, 0.3158]),
    #     np.array([0, 0.7071068, 0.7071068, 0]),
    # )

    # above_cup_pose = (
    #     np.array([0.54, 0.58, 0.4158]),
    #     np.array([0, 0.7071068, 0.7071068, 0]),
    # )
    # above_cup_pos = [3.141430200763298, -1.4221115105035542, -1.7185037629661792, -1.1659074241546614, 0.34054286938325806, -0.440465006378167, 2.692516315826855]

    # before_transfer_pos = [-2.86554642, -1.61951779, -2.60986085, -1.37302839, 1.11779249, -1.18028264, 2.05515862]

    # new values
    # outside_cup_pose = (
    #     np.array([0.545, 0.45, 0.270]),
    #     np.array([-0.2126311, -0.6743797, -0.6743797, 0.2126311]),
    # )
    # outside_cup_pos = [-3.100185292329023, -1.0924888665911388, -0.5706994426374399, -1.424560020809773, -1.4250553687725285, -1.041275746196697, -2.8561579774322996]

    # cup_inside_mount = (
    #     np.array([0.545, 0.518, 0.270]),
    #     np.array([-0.2126311, -0.6743797, -0.6743797, 0.2126311]),
    # )
    # cup_inside_mount_pos = [3.042634381172411, -1.168988168903665, -0.5663478374162505, -1.2153447487342381, -1.3638740364179194, -1.0536210957458687, -2.9810178882956833]

    # above_cup_pose = (
    #     np.array([0.545, 0.518, 0.370]),
    #     np.array([-0.2126311, -0.6743797, -0.6743797, 0.2126311]),
    # )
    # above_cup_pos = [3.0622933037071576, -0.9648787092700299, -0.5952463310369183, -1.2963117700914815, -1.4352504820575698, -0.9462605500892867, -3.085153612188289]

    cup_outside_mount: Pose = Pose(
        (0.545, 0.45, 0.270),
        (-0.2126311, -0.6743797, -0.6743797, 0.2126311),
    )
    cup_outside_mount_pos: JointPositions = field(
        default_factory=lambda: [
            -3.100185292329023,
            -1.0924888665911388,
            -0.5706994426374399,
            -1.424560020809773,
            -1.4250553687725285,
            -1.041275746196697,
            -2.8561579774322996,
        ]
    )

    cup_inside_mount: Pose = Pose(
        (0.545, 0.518, 0.270),
        (-0.2126311, -0.6743797, -0.6743797, 0.2126311),
    )
    cup_inside_mount_pos: JointPositions = field(
        default_factory=lambda: [
            3.042634381172411,
            -1.168988168903665,
            -0.5663478374162505,
            -1.2153447487342381,
            -1.3638740364179194,
            -1.0536210957458687,
            -2.9810178882956833,
        ]
    )

    cup_above_mount: Pose = Pose(
        (0.545, 0.518, 0.370),
        (-0.2126311, -0.6743797, -0.6743797, 0.2126311),
    )
    cup_above_mount_pos: JointPositions = field(
        default_factory=lambda: [
            3.0622933037071576,
            -0.9648787092700299,
            -0.5952463310369183,
            -1.2963117700914815,
            -1.4352504820575698,
            -0.9462605500892867,
            -3.085153612188289,
        ]
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
    wiper_transfer_transform: Pose = Pose(
        (-0.1, 0.35, 0.0),
        p.getQuaternionFromEuler((-np.pi / 2, np.pi, np.pi / 2)),
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
    utensil_corner_waypoint_transform: Pose = Pose((0.1, -0.4, -0.3))
    # This is relative to the wheelchair head.
    utensil_staging_transform: Pose = Pose(
        (0.0, 0.75, 0.0), p.getQuaternionFromEuler((-np.pi / 2, np.pi, np.pi / 2))
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
    def wiper_transfer_pose(self) -> Pose:
        """Pose for the finger tip for wiper transfer."""
        target_wiper_pose = multiply_poses(
            self.wheelchair_head_pose, self.wiper_transfer_transform
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
    def utensil_corner_waypoint_pose(self) -> Pose:
        """Pose for the finger tip for the corner waypoint for the utensil."""
        return multiply_poses(
            self.utensil_pose,
            self.utensil_corner_waypoint_transform,
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
            "outer_camera_yaw": 30,
            "outer_camera_distance": 2.5,
            "outer_camera_pitch": -30,
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
