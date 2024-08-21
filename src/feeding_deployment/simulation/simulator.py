"""A PyBullet-based simulator for the feeding deployment environment."""

from __future__ import annotations

import pybullet as p
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.single_arm import SingleArmTwoFingerGripperPyBulletRobot
from pybullet_helpers.utils import create_pybullet_block
from feeding_deployment.simulation.scene_description import SceneDescription
from pybullet_helpers.gui import create_gui_connection
from feeding_deployment.simulation.state import FeedingDeploymentSimulatorState
from pybullet_helpers.inverse_kinematics import (
    set_robot_joints_with_held_object,
)

class FeedingDeploymentPyBulletSimulator:
    """A PyBullet-based simulator for the feeding deployment environment."""

    def __init__(self, scene_description: SceneDescription) -> None:
        self.scene_description = scene_description

        # Create the PyBullet client.
        self.physics_client_id = create_gui_connection(camera_yaw=180)

        # Create robot.
        self.robot: SingleArmTwoFingerGripperPyBulletRobot = create_pybullet_robot(
            scene_description.robot_name,
            self.physics_client_id,
            base_pose=scene_description.robot_base_pose,
            control_mode="reset",
            home_joint_positions=scene_description.initial_joints,
        )

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
        cup_collision_id = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=scene_description.cup_radius,
            height=scene_description.cup_length,
            physicsClientId=self.physics_client_id,
        )
        cup_visual_id = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=scene_description.cup_radius,
            length=scene_description.cup_length,
            rgbaColor=scene_description.cup_rgba,
            physicsClientId=self.physics_client_id,
        )
        cup_handle_collision_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=scene_description.cup_handle_half_extents,
            physicsClientId=self.physics_client_id,
        )
        cup_handle_visual_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=scene_description.cup_handle_half_extents,
            rgbaColor=scene_description.cup_handle_rgba,
            physicsClientId=self.physics_client_id,
        )
        self.cup_id = p.createMultiBody(
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
            physicsClientId=self.physics_client_id,
        )
        self.cup_handle_link_id = 0

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
        self._held_object_name: str | None = None

    def get_collision_ids(self) -> set[int]:
        """Return all collision IDs."""
        collision_ids = {
            self._table_id,
            self.robot_holder_id,
            self._wheelchair_id,
            self.cup_id,
        }
        if self._held_object_name == "cup":
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
        else:
            assert state.held_object == "cup"
            set_robot_joints_with_held_object(
                self.robot, self.physics_client_id, self.cup_id,
                state.held_object_tf, state.robot_joints,
            )

