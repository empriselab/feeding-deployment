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

    # # Constants across all tools
    # above_plate_pos: JointPositions = field(
    #     default_factory=lambda: [
    #         -2.86495014,
    #         -1.61460533,
    #         -2.6115943,
    #         -1.37673391,
    #         1.11842806,
    #         -1.17904586,
    #         -2.6957422,
    #     ]
    # )

    above_plate_pos: JointPositions = field(
        default_factory=lambda: [
            -2.24219867867983, 
            -1.117632303074795, 
            -1.822950011441888, 
            -2.108239012687754, 
            -0.6023660258951367, 
            -0.27174949026390305, 
            -1.4889110474937937
        ]
    )

    # before_transfer_pos: JointPositions = field(
    #     default_factory=lambda: [
    #         -2.86554642,
    #         -1.61951779,
    #         -2.60986085,
    #         -1.37302839,
    #         1.11779249,
    #         -1.18028264,
    #         2.05515862,
    #     ]
    # )

    before_transfer_pos: JointPositions = field(
        default_factory=lambda: [
            -2.291869562487007, 
            -1.3006196994707935, 
            -1.720891553199544, 
            -2.208449769765801, 
            -0.477821699620927, 
            -0.1613864967943659, 
            -2.9981518678009262
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
        (-0.286, -0.133, 0.065),
        (0.7071068, -0.7071068, 0, 0),
    )
    utensil_inside_mount_pos: JointPositions = field(
        default_factory=lambda: [
            2.2467215251591117, 
            0.5836147086598885, 
            -2.535738152982762, 
            -2.2963045246503158, 
            -0.7542663428866403, 
            -0.4699185027700077, 
            1.9096838418235886
        ]
    )

    utensil_outside_mount: Pose = Pose(
        (-0.286, -0.263, 0.065),
        (0.7071068, -0.7071068, 0, 0),
    )
    utensil_outside_mount_pos: JointPositions = field(
        default_factory=lambda: [
            1.9202113176126718, 
            0.6917033625584985, 
            -2.4682459277929927, 
            -2.063521874824234, 
            -0.7181986196114787, 
            -0.6391235078926281, 
            1.523472620014461
        ]
    )

    utensil_above_mount: Pose = Pose(
        (-0.286, -0.133, 0.165),
        (0.7071068, -0.7071068, 0, 0),
    )
    utensil_above_mount_pos: JointPositions = field(
        default_factory=lambda: [
            1.9755942824839052, 
            0.3527602540420511, 
            -2.2270956258218817, 
            -2.1930039680755593, 
            -0.39397207269978196, 
            -0.7730064748453094, 
            1.5904074455880766
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

    # # Constants for drink pick and place
    # drink_gaze_pos: JointPositions = field(
    #     default_factory=lambda: [
    #         -0.004187021865822871, 0.6034579885210962, -3.1259047705564633, -2.3538005746884725, 0.01149092320739253, 1.3411586039000891, 1.6825233913747728
    #     ]
    # )

    drink_gaze_pos: JointPositions = field(
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

    drink_staging_pos: JointPositions = field(
        default_factory=lambda: [
            # -2.66714970644385, -1.1667276777704059, -0.9741023013894106, -1.4808070482966826, -0.9401592480319145, -0.8664637217150242, -2.4832424542073377
            # -0.017759556045302105, 0.694072510664233, 3.122327878896378, -2.0030297685597382, 1.3270309337913822, 1.3287449442689572, 1.1063354400467882
            -2.5860902733967808, -1.105096803823792, -1.0315333702969696, -1.3979449215077393, -0.7852325147776451, -0.8370922506847585, -2.7182634909296315,
        ]
    )

    # drink_pre_staging_pos: JointPositions = field(
    #     default_factory=lambda: [
    #         -2.4297866858589403, -1.1346866540634446, -0.9258335718961606, -1.5283066568671186, -0.8368781325331156, -1.0769876711502242, -2.506139514311512
    #     ]
    # )

    # drink_staging_pos: JointPositions = field(
    #     default_factory=lambda: [
    #        -2.7565158052469845, -1.184186829244747, -1.0646716163730439, -1.6191545385007728, -1.093139210529337, -0.6452967152994313, -2.309888776738592
    #     ]
    # )

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
        (-0.444, -0.139, 0.065),
        (0.7071068, -0.7071068, 0, 0),
    )
    wipe_inside_mount_pos: JointPositions = field(
        default_factory=lambda: [
            2.4024543972753483, 
            0.7942983146024386, 
            -2.472859055433228, 
            -1.8032721804634013, 
            -0.6638818512826585, 
            -0.7855201361753741, 
            1.851469670397492
        ]
    )

    wipe_outside_mount: Pose = Pose(
        (-0.444, -0.269, 0.065),
        (0.7071068, -0.7071068, 0, 0),
    )
    wipe_outside_mount_pos: JointPositions = field(
        default_factory=lambda: [
            2.188196829149912, 
            0.882908608288209, 
            -2.4520760126023586, 
            -1.6022200297611775, 
            -0.6621619818506854, 
            -0.905755467805835, 
            1.5558217714595446
        ]
    )

    wipe_above_mount: Pose = Pose(
        (-0.444, -0.139, 0.165),
        (0.7071068, -0.7071068, 0, 0),
    )
    wipe_above_mount_pos: JointPositions = field(
        default_factory=lambda: [
            2.323328152863568, 
            0.6356845013979816, 
            -2.3499581658677062, 
            -1.7172542077827142, 
            -0.5122867325520781, 
            -1.011576173720134, 
            1.7346615607738536
        ]
    )

    wipe_outside_above_mount: Pose = Pose(
        (-0.444, -0.269, 0.165),
        (0.7071068, -0.7071068, 0, 0),
    )

    wipe_outside_above_mount_pos: JointPositions = field(
        default_factory=lambda: [
            2.141991250553146, 
            0.7403159727275052, 
            -2.361341048998923, 
            -1.5166649144779463, 
            -0.546357617641898, 
            -1.1156594333392649, 
            1.4704801085376829
        ]
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
