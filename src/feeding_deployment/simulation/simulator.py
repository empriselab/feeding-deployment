"""A PyBullet-based simulator for the feeding deployment environment."""

from __future__ import annotations

import pybullet as p
from pybullet_helpers.geometry import Pose
from pybullet_helpers.gui import create_gui_connection
from pybullet_helpers.inverse_kinematics import (
    set_robot_joints_with_held_object,
)
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot
from pybullet_helpers.utils import create_pybullet_block

from feeding_deployment.simulation.scene_description import SceneDescription
from feeding_deployment.simulation.state import FeedingDeploymentSimulatorState


class FeedingDeploymentPyBulletSimulator:
    """A PyBullet-based simulator for the feeding deployment environment."""

    def __init__(self, scene_description: SceneDescription, use_gui: bool = True, ignore_user = False) -> None:
        self.scene_description = scene_description

        # Create the PyBullet client.
        if use_gui:
            self.physics_client_id = create_gui_connection(camera_yaw=180)
        else:
            self.physics_client_id = p.connect(p.DIRECT)

        # Create robot.
        robot = create_pybullet_robot(
            scene_description.robot_name,
            self.physics_client_id,
            base_pose=scene_description.robot_base_pose,
            control_mode="reset",
            home_joint_positions=scene_description.initial_joints,
        )
        assert isinstance(robot, FingeredSingleArmPyBulletRobot)
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

        if not ignore_user:
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

            # Create a conservative collision boundary around the wheelchair.
            self.conservative_bb_id = create_pybullet_block(
                scene_description.conservative_bb_rgba,
                half_extents=scene_description.conservative_bb_half_extents,
                physics_client_id=self.physics_client_id,
            )
            p.resetBasePositionAndOrientation(
                self.conservative_bb_id,
                scene_description.conservative_bb_pose.position,
                scene_description.conservative_bb_pose.orientation,
                physicsClientId=self.physics_client_id,
            )
        else:
            self._wheelchair_id = None
            self.conservative_bb_id = None

        # Create drink.
        self.drink_id = p.loadURDF(
            str(scene_description.drink_urdf_path),
            useFixedBase=True,
            physicsClientId=self.physics_client_id,
        )
        p.resetBasePositionAndOrientation(
            self.drink_id,
            scene_description.drink_pose.position,
            scene_description.drink_pose.orientation,
            physicsClientId=self.physics_client_id,
        )

        # Create table.
        self.table_id = create_pybullet_block(
            scene_description.table_rgba,
            half_extents=scene_description.table_half_extents,
            physics_client_id=self.physics_client_id,
        )
        p.resetBasePositionAndOrientation(
            self.table_id,
            scene_description.table_pose.position,
            scene_description.table_pose.orientation,
            physicsClientId=self.physics_client_id,
        )

        # Create wipe.
        self.wipe_id = p.loadURDF(
            str(scene_description.wipe_urdf_path),
            useFixedBase=True,
            physicsClientId=self.physics_client_id,
        )
        p.resetBasePositionAndOrientation(
            self.wipe_id,
            scene_description.wipe_pose.position,
            scene_description.wipe_pose.orientation,
            physicsClientId=self.physics_client_id,
        )

        # Create feeding utensil.
        self.utensil_id = p.loadURDF(
            str(scene_description.utensil_urdf_path),
            useFixedBase=True,
            physicsClientId=self.physics_client_id,
        )
        p.resetBasePositionAndOrientation(
            self.utensil_id,
            scene_description.utensil_pose.position,
            scene_description.utensil_pose.orientation,
            physicsClientId=self.physics_client_id,
        )

        # Track held objects.
        self.held_object_name: str | None = None
        self.held_object_id: int | None = None
        self.held_object_tf: Pose | None = None

    def get_collision_ids(self) -> set[int]:
        """Return all collision IDs."""
        collision_ids = {
            self.table_id,
            self.conservative_bb_id,
            self.robot_holder_id,
            self._wheelchair_id,
            self.drink_id,
            self.wipe_id,
            self.utensil_id,
        }

        # filter none values (wheelchair and conservative_bb_id when ignore_user is True)
        collision_ids = {x for x in collision_ids if x is not None}

        if self.held_object_name == "drink":
            collision_ids.remove(self.drink_id)
        if self.held_object_name == "wipe":
            collision_ids.remove(self.wipe_id)
        if self.held_object_name == "utensil":
            collision_ids.remove(self.utensil_id)
        return collision_ids

    def sync(self, state: FeedingDeploymentSimulatorState) -> None:
        """Sync the simulator to a given state."""
        self.robot.set_joints(state.robot_joints)

        some_object_held = False
        for obj_id, name, state_pose in [
            (self.drink_id, "drink", state.drink_pose),
            (self.wipe_id, "wipe", state.wipe_pose),
            (self.utensil_id, "utensil", state.utensil_pose),
        ]:
            if state.held_object == name:
                assert not some_object_held
                some_object_held = True
                self.held_object_name = name
                self.held_object_id = obj_id
                self.held_object_tf = state.held_object_tf
                set_robot_joints_with_held_object(
                    self.robot,
                    self.physics_client_id,
                    obj_id,
                    state.held_object_tf,
                    state.robot_joints,
                )
            else:
                assert state_pose is not None
                p.resetBasePositionAndOrientation(
                    obj_id,
                    state_pose.position,
                    state_pose.orientation,
                    physicsClientId=self.physics_client_id,
                )
        if not some_object_held:
            self.held_object_name = None
            self.held_object_id = None
            self.held_object_tf = None
