"""A PyBullet-based simulator for the feeding deployment environment."""

from __future__ import annotations

import time
import numpy as np
from pathlib import Path
import imageio.v2 as iio

import pybullet as p
from pybullet_helpers.geometry import Pose
from pybullet_helpers.gui import create_gui_connection
from pybullet_helpers.inverse_kinematics import set_robot_joints_with_held_object
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot
from pybullet_helpers.utils import create_pybullet_block
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.gui import visualize_pose
from pybullet_helpers.camera import capture_superimposed_image
from pybullet_helpers.inverse_kinematics import add_fingers_to_joint_positions
from pybullet_helpers.motion_planning import run_motion_planning

from feeding_deployment.simulation.scene_description import SceneDescription
from feeding_deployment.simulation.state import FeedingDeploymentSimulatorState
from feeding_deployment.simulation.planning import (
    _plan_to_sim_state_trajectory,
    remap_trajectory_to_constant_distance,
    _get_plan_to_execute_grasp,
    _get_plan_to_execute_ungrasp,
)
from feeding_deployment.simulation.control import cartesian_control_step


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

        self._user_head = p.loadURDF(
            str(scene_description.user_head_urdf_path),
            useFixedBase=True,
            physicsClientId=self.physics_client_id,
        )
        p.resetBasePositionAndOrientation(
            self._user_head,
            scene_description.user_head_pose.position,
            scene_description.user_head_pose.orientation,
            physicsClientId=self.physics_client_id,
        )

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
        self.utensil_joints = []
        for i in range(p.getNumJoints(self.utensil_id)):
            joint_info = p.getJointInfo(self.utensil_id, i)
            if joint_info[2] != 4: # Skip fixed joints.
                self.utensil_joints.append(i)

        # Track held objects.
        self.held_object_name: str | None = None
        self.held_object_id: int | None = None
        self.held_object_tf: Pose | None = None

        self.recorded_states: list[FeedingDeploymentSimulatorState] = []

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
    
    def set_robot_motors(self, target_positions: list[float]) -> None:
        """Move the robot to a given state."""
        self.robot.set_motors(target_positions)
        p.stepSimulation(physicsClientId=self.physics_client_id)
        # Rajat TODO: Update all the other objects in the scene as well.
    
    def set_utensil_motors(self, target_positions: list[float]) -> None:
        """Move the utensil to a given state."""
        assert len(target_positions) == len(self.utensil_joints)

        p.setJointMotorControlArray(
            bodyUniqueId=self.utensil_id,
            jointIndices=self.utensil_joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_positions,
            physicsClientId=self.physics_client_id,
        )
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.physics_client_id)

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

    def move_to_ee_pose(self, pose: Pose, max_control_time: float = 30.0) -> list[FeedingDeploymentSimulatorState]:
        """Move the robot to the specified end effector pose using cartesian control."""

        visualize_pose(pose, self.physics_client_id)
        visualize_pose(self.robot.get_end_effector_pose(), self.physics_client_id)
    
        joint_trajectory: list[JointPositions] = []
            
        start_time = time.time()
        target_reached = False
        while time.time() - start_time < max_control_time:
            current_pose = self.robot.get_end_effector_pose()
            if pose.allclose(current_pose, atol=1e-2):
                target_reached = True
                break
            current_joint_positions = self.robot.get_joint_positions()
            joint_trajectory.append(current_joint_positions)
            current_jacobian = self.robot.get_jacobian()
            target_positions = cartesian_control_step(current_joint_positions, current_jacobian, current_pose, pose)
            target_positions = np.concatenate((target_positions, self.robot.finger_state_to_joints(self.scene_description.tool_grasp_fingers_value))) # Rajat ToDo: Remove hardcoding
            self.set_robot_motors(target_positions)
        
        if not target_reached:
            raise RuntimeError("Sim cartesian controller: Failed to reach target pose in time")

        plan = _plan_to_sim_state_trajectory(joint_trajectory, self)
        plan_states = remap_trajectory_to_constant_distance(plan, self)
        self.recorded_states.extend(plan_states)

        return plan_states

    def move_to_joint_positions(self, joint_positions: list[float], max_control_time: float = 30.0) -> list[FeedingDeploymentSimulatorState]:
        """Move the robot to the specified joint positions."""
        
        initial_joint_positions = self.robot.get_joint_positions().copy()
        target_joint_positions = add_fingers_to_joint_positions(sim.robot, joint_positions)

        direct_path = run_motion_planning(
            robot=self.robot,
            initial_positions=initial_joint_positions,
            target_positions=target_joint_positions,
            collision_bodies=self.get_collision_ids(),
            seed=0,  # not used
            physics_client_id=self.physics_client_id,
            held_object=self.held_object_id,
            base_link_to_held_obj=self.held_object_tf,
            direct_path_only=True,
        )

        if direct_path:
            # Rajat ToDo: Discuss arm / robot dissociation with Tom
            plan = _plan_to_sim_state_trajectory(direct_path, self)
        else:
            raise NotImplementedError("No direct path found. But motion planning is not implemented yet.")
            print("No direct path found. Running motion planning.")
            plan = run_motion_planning(
                robot=sim.robot,
                initial_positions=initial_joint_positions,
                target_positions=target_joint_positions,
                collision_bodies=sim.get_collision_ids(),
                seed=0,
                physics_client_id=sim.physics_client_id,
                held_object=sim.held_object_id,
                base_link_to_held_obj=sim.held_object_tf,
            )
            plan = _plan_to_sim_state_trajectory(plan, sim)
            plan = remap_trajectory_to_constant_distance(plan, sim)
            robot_commands.extend(simulated_trajectory_to_kinova_commands(plan))
        
        plan_states = remap_trajectory_to_constant_distance(plan, self)
        self.recorded_states.extend(plan_states)

    def grasp_object(self, object_name: str) -> list[FeedingDeploymentSimulatorState]:
        plan = _get_plan_to_execute_grasp(self, object_name)
        self.recorded_states.extend(plan)

    def ungrasp_object(self) -> list[FeedingDeploymentSimulatorState]:
        plan = _get_plan_to_execute_ungrasp(self)
        self.recorded_states.extend(plan)

    def close_gripper(self) -> None:
        raise NotImplementedError("TODO")
        self.robot.close_fingers()

    def open_gripper(self) -> None:
        raise NotImplementedError("TODO")
        self.robot.open_fingers()

    def make_simulation_video(self, outfile: Path, fps: int = 20) -> None:
        """Make a video for a simulated drink manipulation plan."""
        imgs = []
        for state in self.recorded_states:
            self.sync(state)
            img = capture_superimposed_image(
                self.physics_client_id, **self.scene_description.camera_kwargs
            )
            imgs.append(img)
        iio.mimsave(outfile, imgs, fps=fps)  # type: ignore
        print(f"Wrote out to {outfile}")