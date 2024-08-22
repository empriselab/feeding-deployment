"""A PyBullet-based simulator for the feeding deployment environment."""

from __future__ import annotations

import pybullet as p
from pybullet_helpers.geometry import Pose
from pybullet_helpers.gui import create_gui_connection
from pybullet_helpers.inverse_kinematics import (
    set_robot_joints_with_held_object,
)
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.single_arm import SingleArmTwoFingerGripperPyBulletRobot
from pybullet_helpers.utils import create_pybullet_block

from feeding_deployment.simulation.scene_description import SceneDescription
from feeding_deployment.simulation.state import FeedingDeploymentSimulatorState


class FeedingDeploymentPyBulletSimulator:
    """A PyBullet-based simulator for the feeding deployment environment."""

    def __init__(self, scene_description: SceneDescription) -> None:
        self.scene_description = scene_description

        # Create the PyBullet client.
        self.physics_client_id = create_gui_connection(camera_yaw=180)

        # Create robot.
        robot = create_pybullet_robot(
            scene_description.robot_name,
            self.physics_client_id,
            base_pose=scene_description.robot_base_pose,
            control_mode="reset",
            home_joint_positions=scene_description.initial_joints,
        )
        assert isinstance(robot, SingleArmTwoFingerGripperPyBulletRobot)
        robot.close_fingers()
        self.robot = robot

        # Create a holder (vention stand).
        self.robot_holder_id = create_pybullet_block(
            scene_description.robot_holder_rgba,
            half_extents=scene_description.robot_holder_half_extents,
            physics_client_id=self.physics_client_id,
        )
        p.resetBasePositionAndOrientation(
            self.robot_holder_id,
            scene_description.robot_holder_pose.position,
            scene_description.robot_holder_pose.orientation,
            physicsClientId=self.physics_client_id,
        )

        # Create wheelchair.
        self._wheelchair_id = p.loadURDF(
            str(scene_description.wheelchair_urdf_path),
            useFixedBase=True,
            physicsClientId=self.physics_client_id,
        )
        p.resetBasePositionAndOrientation(
            self._wheelchair_id,
            scene_description.wheelchair_pose.position,
            scene_description.wheelchair_pose.orientation,
            physicsClientId=self.physics_client_id,
        )

        # Create cup.
        self.cup_id = p.loadURDF(
            str(scene_description.cup_urdf_path),
            useFixedBase=True,
            physicsClientId=self.physics_client_id,
        )
        p.resetBasePositionAndOrientation(
            self.cup_id,
            scene_description.cup_pose.position,
            scene_description.cup_pose.orientation,
            physicsClientId=self.physics_client_id,
        )
        # Create a table.
        self._table_id = create_pybullet_block(
            scene_description.table_rgba,
            half_extents=scene_description.table_half_extents,
            physics_client_id=self.physics_client_id,
        )
        p.resetBasePositionAndOrientation(
            self._table_id,
            scene_description.table_pose.position,
            scene_description.table_pose.orientation,
            physicsClientId=self.physics_client_id,
        )

        # Track held objects.
        self.held_object_name: str | None = None
        self.held_object_id: int | None = None
        self.held_object_tf: Pose | None = None

    def get_collision_ids(self) -> set[int]:
        """Return all collision IDs."""
        collision_ids = {
            self._table_id,
            self.robot_holder_id,
            self._wheelchair_id,
            self.cup_id,
        }
        if self.held_object_name == "cup":
            collision_ids.remove(self.cup_id)
        return collision_ids

    def sync(self, state: FeedingDeploymentSimulatorState) -> None:
        """Sync the simulator to a given state."""
        self.robot.set_joints(state.robot_joints)
        if state.cup_pose is not None:
            p.resetBasePositionAndOrientation(
                self.cup_id,
                state.cup_pose.position,
                state.cup_pose.orientation,
                physicsClientId=self.physics_client_id,
            )
            self.held_object_name = None
            self.held_object_id = None
            self.held_object_tf = None
        else:
            assert state.held_object == "cup"
            self.held_object_name = state.held_object
            self.held_object_id = self.cup_id
            self.held_object_tf = state.held_object_tf
            set_robot_joints_with_held_object(
                self.robot,
                self.physics_client_id,
                self.cup_id,
                state.held_object_tf,
                state.robot_joints,
            )
