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

    # Robot.
    robot_name: str = "kinova-gen3"
    initial_joints: JointPositions = field(
        default_factory=lambda: [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0

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

    # Constants across all tools
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

    tool_grasp_fingers_value: float = 0.44

    ######### Utensil #########

    # Used by simulator to spawn the utensil
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

    # Constants for utensil pick and place
    utensil_inside_mount: Pose = Pose(
        (0.242, -0.077, 0.07),
        (-1, 0, 0, 0),
    )
    utensil_inside_mount_pos: JointPositions = field(
        default_factory=lambda: [
            0.03610898146611135, 
            0.46058565446690675, 
            -2.7959276599029583, 
            -2.4802831496040416, 
            -0.5912420020218754, 
            -0.27659484555119374, 
            0.9346522303717946,
        ]
    )

    utensil_outside_mount: Pose = Pose(
        (0.372, -0.077, 0.07),
        (-1, 0, 0, 0),
    )
    utensil_outside_mount_pos: JointPositions = field(
        default_factory=lambda: [
            -0.13616795916796942, 0.6152003983994736, -2.673508269862523, -2.09060888627143, -0.5157179493005364, -0.5540355110645967, 0.7160963868259976,
        ]
    )

    utensil_above_mount: Pose = Pose(
        (0.242, -0.077, 0.17),
        (-1, 0, 0, 0),
    )
    utensil_above_mount_pos: JointPositions = field(
        default_factory=lambda: [
            -0.3081224117999879, 
            0.1449308244187662, 
            -2.4515079603418446, 
            -2.3539334664268674, 
            -0.14376009880356744, 
            -0.6872590793313744, 
            0.5028097739444904,
        ]
    )

    utensil_outside_above_mount: Pose = Pose(
        (0.372, -0.077, 0.17),
        (-1, 0, 0, 0),
    )
    utensil_outside_above_mount_pos: JointPositions = field(
        default_factory=lambda: [
            -0.2692035082617874, 0.4127082432063301, -2.513398492494741, -1.9930522355357558, -0.31928105676741936, -0.8392446174777604, 0.5472652562309106,
        ]
    )

    # Constants for utensil transfer
    # Rajat ToDo: Fix with correct values, copied from drinking utensil
    utensil_tip_from_end_effector: Pose = Pose(
        (0.255, 0.0, -0.018),
        (0.000, 0.707, 0.000, 0.707),
    )

    ######### drink #########

    drink_pose: Pose = Pose(
        (0.545, 0.65, 0.270), 
        (-0.2126311, -0.6743797, -0.6743797, 0.2126311)
    )
    drink_urdf_path: Path = (
        Path(__file__).parent.parent
        / "assets"
        / "urdf"
        / "drinking_utensil"
        / "drinking_utensil.urdf"
    )

    # Constants for drink pick and place
    drink_outside_mount: Pose = Pose(
        (0.55, 0.46, 0.305),
        (-0.2126311, -0.6743797, -0.6743797, 0.2126311),
    )
    drink_outside_mount_pos: JointPositions = field(
        default_factory=lambda: [
            -2.9457621628368873, -1.206488672845289, -1.0073524002312677, -1.3997867637176382, -1.0606635589324744, -0.7359768177844117, -3.048252585808042
        ]
    )

    drink_inside_mount: Pose = Pose(
        (0.55, 0.52, 0.305),
        (-0.2126311, -0.6743797, -0.6743797, 0.2126311),
    )
    drink_inside_mount_pos: JointPositions = field(
        default_factory=lambda: [
            -3.0706449768856463, -1.233024942579057, -1.0107718990709298, -1.3064307169693468, -1.0801286033398636, -0.6790118020676168, -3.0605814237584545
        ]
    )

    drink_above_mount: Pose = Pose(
        (0.55, 0.52, 0.405),
        (-0.2126311, -0.6743797, -0.6743797, 0.2126311),
    )
    drink_above_mount_pos: JointPositions = field(
        default_factory=lambda: [
            -3.0822113518159693, -1.0554986243143745, -0.9943061066831875, -1.2514305815048123, -1.0634620086059297, -0.7145069457084112, 3.0023889570952424
        ]
    )

    # Constants for drink transfer
    drink_tip_from_end_effector: Pose = Pose(
        (0.270, 0.095, -0.002),
        (-0.000, 0.707, 0.000, 0.707),
    )

    ######### Wiping Utensil #########

    wipe_pose: Pose = Pose(
        (0.35, 0.15, -0.05), p.getQuaternionFromEuler((0.0, np.pi, np.pi))
    )
    wipe_urdf_path: Path = (
        Path(__file__).parent.parent
        / "assets"
        / "urdf"
        / "wiping_utensil"
        / "wiping_utensil.urdf"
    )

    # Constants for wiping utensil pick and place
    wipe_inside_mount: Pose = Pose(
        (-0.147, -0.17, 0.07),
        (0.7071068, -0.7071068, 0.0, 0.0),
    )
    wipe_inside_mount_pos: JointPositions = field(
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

    wipe_outside_mount: Pose = Pose(
        (-0.147, -0.29, 0.07),
        (0.7071068, -0.7071068, 0.0, 0.0),
    )
    wipe_outside_mount_pos: JointPositions = field(
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

    wipe_above_mount: Pose = Pose(
        (-0.147, -0.17, 0.15),
        (0.7071068, -0.7071068, 0.0, 0.0),
    )
    wipe_above_mount_pos: JointPositions = field(
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

    wipe_infront_mount: Pose = Pose(
        (0.0, -0.17, 0.15),
        (0.7071068, -0.7071068, 0.0, 0.0),
    )
    wipe_infront_mount_pos: JointPositions = field(
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

    wipe_neutral_pos: JointPositions = field(
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

    # Constants for wiping utensil transfer
    # Rajat ToDo: Fix with correct values, copied from drinking utensil
    wipe_tip_from_end_effector: Pose = Pose(
        (0.270, 0.095, -0.002),
        (-0.000, 0.707, 0.000, 0.707),
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
            "drink_pose",
            "wipe_pose",
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
