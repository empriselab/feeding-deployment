"""The description of a simulation initial state, with default values."""

from __future__ import annotations

import json
import pickle
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

import yaml

def create_scene_description_from_config(config_file_path: str, transfer_type: str) -> SceneDescription:
    """Create a SceneDescription instance from a YAML configuration file."""
    # Load the YAML file
    with open(config_file_path, "r") as file:
        config = yaml.safe_load(file)

    # Process the configuration dictionary
    processed_config = {}
    for key, value in config.items():
        if isinstance(value, dict):
            value_type = value.get("type")
            values = value.get("values")

            if not value_type or values is None:
                raise ValueError(f"Key '{key}' is missing 'type' or 'values': {value}")

            if value_type == "joint_positions":
                # Handle joint positions
                processed_config[key] = values
            elif value_type == "ee_pose":
                # Handle end-effector poses
                if len(values) != 7:
                    raise ValueError(f"Pose for key '{key}' must have 7 values (3 position, 4 quaternion), got {len(values)}")
                position = tuple(values[:3])
                orientation = tuple(values[3:])
                processed_config[key] = Pose(position, orientation)
            else:
                raise ValueError(f"Unknown type '{value_type}' for key '{key}'")
        else:
            raise ValueError(f"Unexpected value type for key '{key}': {type(value)}")
    
    processed_config["transfer_type"] = transfer_type
    processed_config["scene_label"] = Path(config_file_path).stem

    # Create an instance of SceneDescription using the processed config
    return SceneDescription(**processed_config)


@dataclass(frozen=False)
class SceneDescription:
    """Scene description."""

    scene_label: str
    transfer_type: str

    # Robot constants
    initial_joints: JointPositions
    retract_pos: JointPositions
    retract_utensil_forward_pos: JointPositions

    # Feeding task constants
    plate_gaze_pos: JointPositions
    above_plate_pos: JointPositions
    before_transfer_pos: JointPositions
    drink_before_transfer_pos: JointPositions
    absolute_before_transfer_pos: JointPositions
    before_transfer_pose: Pose
    drink_before_transfer_pose: Pose

    # Utensil mount constants
    utensil_inside_mount: Pose
    utensil_inside_mount_pos: JointPositions
    utensil_outside_mount: Pose
    utensil_outside_mount_pos: JointPositions
    utensil_above_mount: Pose
    utensil_above_mount_pos: JointPositions

    # Drink placement constants
    drink_gaze_pos: JointPositions
    drink_staging_pos: JointPositions

    # Wipe mount constants
    wipe_inside_mount: Pose
    wipe_inside_mount_pos: JointPositions
    wipe_outside_mount: Pose
    wipe_outside_mount_pos: JointPositions
    wipe_above_mount: Pose
    wipe_above_mount_pos: JointPositions

    # Plate constants
    plate_staging_pos: JointPositions
    plate_delta_xy: tuple[float, float] = (0.0, 0.0)
    drink_delta_xy: tuple[float, float] = (0.0, 0.0)

    # Specific env arguments can be set to None
    utensil_outside_above_mount: Pose = None
    utensil_outside_above_mount_pos: JointPositions = None
    # Used for specific environments
    wipe_outside_above_mount: Pose = None
    wipe_outside_above_mount_pos: JointPositions = None
    wipe_infront_mount: Pose = None 
    wipe_infront_mount_pos: JointPositions = None
    wipe_neutral_pos: JointPositions = None

    # Robot.
    robot_name: str = "kinova-gen3"
    robot_urdf_path: Path = (Path(__file__).parent.parent / "assets" / "robot" / "robot.urdf")
    robot_base_pose: Pose = Pose(
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    tool_frame_to_finger_tip: Pose = Pose(
        (0.0, 0.0, 0.05955),
        (0.0, 0.0, 0.0, 1.0),
    )
    # end_effector_link to camera_color_optical_frame
    camera_pose: Pose = Pose(
        (-0.046, 0.083, 0.125),
        (0.006, 0.708, 0.005, 0.706),
    )

    
    # - Translation: [-0.046, 0.084, 0.125]
    # - Rotation: in Quaternion [0.001, 0.707, -0.002, 0.707]
    #             in RPY (radian) [-0.749, 1.569, -0.753]
    #             in RPY (degree) [-42.905, 89.924, -43.172]


    # Robot holder (vention stand).
    # robot_holder_pose: Pose = Pose((0.0, 0.0, -0.261))
    robot_holder_pose: Pose = Pose((0.0, 0.0, -0.34))
    robot_holder_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    robot_holder_half_extents: tuple[float, float, float] = (0.10, 0.10, 0.33)

    # Wheelchair.
    wheelchair_pose: Pose = Pose(
        (-0.3, 0.45, -0.06), (0.0, 0.0, 0.0, 1.0)
    )
    wheelchair_relative_head_pose: Pose = Pose(
        (0.0, -0.25, 0.75), (0.0, 0.0, 0.0, 1.0)
    )  # Rajat ToDo: Fix this
    wheelchair_urdf_path: Path = (
        Path(__file__).parent.parent
        / "assets"
        / "wheelchair"
        / "wheelchair.urdf"
    )
    wheelchair_mesh_path: Path = (
        Path(__file__).parent.parent
        / "assets"
        / "wheelchair"
        / "wheelchair.obj"
    )
    
    user_head_pose: Pose = Pose(
        (-0.4, 0.5, 0.67), (0.5, 0.5, 0.5, 0.5)
    )

    user_head_urdf_path: Path = (
        Path(__file__).parent.parent
        / "assets"
        / "head_models"
        / "mouth_open.urdf"
    )

    # Conservative bounding box around the wheel chair.
    conservative_bb_pose: Pose = Pose((-0.4, 0.7, -0.25))
    conservative_bb_rgba: tuple[float, float, float, float] = (0.9, 0.1, 0.1, 0.5)
    conservative_bb_half_extents: tuple[float, float, float] = (0.4, 0.4, 1.0)

    # Table.
    table_pose: Pose = Pose((0.35, 0.45, 0.2))
    table_urdf_path: Path = Path(__file__).parent.parent / "assets" / "table" / "table.urdf"
    table_mesh_path: Path = Path(__file__).parent.parent / "assets" / "table" / "table.obj"

    # Floor
    floor_position: tuple[float, float, float] = (0, 0, -0.66)
    floor_urdf: Path = Path(__file__).parent.parent / "assets" / "floor" / "floor.urdf"
    floor_mesh_path: Path = Path(__file__).parent.parent / "assets" / "floor" / "floor.obj"

    wall_poses: list[Pose] = field(
        default_factory=lambda: [
            Pose.from_rpy((0.0, -1.25, 0.0), (np.pi / 2, 0.0, np.pi / 2)),
            Pose.from_rpy((-1.25, 0.0, 0.0), (np.pi / 2, 0.0, 0.0)),
            # Pose.from_rpy((4.25, 0.0, 0.0), (np.pi / 2, 0.0, np.pi)),
            # Pose.from_rpy((0.0, 0.0, 3.0), (0.0, np.pi / 2, 0.0)),
        ]
    )
    wall_half_extents: tuple[float, float, float] = (0.1, 3.0, 5.0)
    wall_texture: Path = Path(__file__).parent.parent / "assets" / "tiled_wall_texture.jpg"

    tool_grasp_fingers_value: float = 0.44

    ######### Simulator Poses for Tools #########

    # Utensil
    utensil_urdf_path: Path = (
        Path(__file__).parent.parent
        / "assets"
        / "feeding_utensil"
        / "feeding_utensil.urdf"
    )
    tool_frame_to_utensil_tip: Pose = Pose(
        (0.255, 0.0, -0.018),
        (0.000, 0.707, 0.000, 0.707),
    )
    
    # Drink
    drink_urdf_path: Path = (
        Path(__file__).parent.parent
        / "assets"
        / "drinking_utensil"
        / "drinking_utensil.urdf"
    )
    tool_frame_to_drink_tip: Pose = Pose(
        (0.210, 0.070, 0.023),
        (0.000, 0.707, 0.000, 0.707),
    )

    # Wipe
    wipe_urdf_path: Path = (
        Path(__file__).parent.parent
        / "assets"
        / "wiping_utensil"
        / "wiping_utensil.urdf"
    )
    tool_frame_to_wipe_tip: Pose = Pose(
        (0.139, -0.015, -0.018),
        (0.000, 0.707, 0.000, 0.707),
    )

    # Plate
    plate_pose: Pose = Pose((0.3, 0.25, 0.16))
    plate_urdf_path: Path = Path(__file__).parent.parent / "assets" / "plate" / "plate.urdf"
    plate_mesh_path: Path = Path(__file__).parent.parent / "assets" / "plate" / "plate.obj"

    @property
    def utensil_pose(self):
        return self.utensil_inside_mount.multiply(self.tool_frame_to_finger_tip)
    
    @property
    def drink_pose(self):
        try:
            with open(Path(__file__).parent.parent / 'integration' / 'log' / 'drink_pickup_pos.pkl', 'rb') as f:
                drink_poses = pickle.load(f)
        except FileNotFoundError:
            with open(Path(__file__).parent.parent / 'integration' / 'log' / 'study_drink_pickup_pos.pkl', 'rb') as f:
                drink_poses = pickle.load(f)
        inside_drink_pose = drink_poses["last_drink_poses"]["inside_top_pose"]
        return inside_drink_pose.multiply(self.tool_frame_to_finger_tip)

    @property
    def wipe_pose(self):
        return self.wipe_inside_mount.multiply(self.tool_frame_to_finger_tip)
    
    # @property
    # def plate_pose(self):
    #     try:
    #         with open(Path(__file__).parent.parent / 'integration' / 'log' / 'plate_pickup_pos.pkl', 'rb') as f:
    #             plate_poses = pickle.load(f)
    #     except FileNotFoundError:
    #         with open(Path(__file__).parent.parent / 'integration' / 'log' / 'study_plate_pickup_pos.pkl', 'rb') as f:
    #             plate_poses = pickle.load(f)
    #     inside_plate_pose = plate_poses["last_plate_poses"]["inside_top_pose"]
    #     return inside_plate_pose.multiply(self.tool_frame_to_finger_tip)

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
            "outer_camera_yaw": 160,
            "outer_camera_distance": 2.5,
            "outer_camera_pitch": -45,
            "outer_image_width": 1000,
            "outer_image_height": 1000,
            "inner_camera_target": head_position,
            "inner_camera_yaw": 0,
            "inner_camera_distance": 1.0,
            "inner_camera_pitch": -20,
        }