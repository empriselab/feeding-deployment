"""A PyBullet-based simulator for the feeding deployment environment."""

from __future__ import annotations

import time
import numpy as np
from pathlib import Path
import imageio.v2 as iio

import pybullet as p
from pybullet_helpers.geometry import Pose, get_pose
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
from pybullet_helpers.link import get_relative_link_pose

from feeding_deployment.simulation.scene_description import SceneDescription
from feeding_deployment.simulation.state import FeedingDeploymentWorldState
from feeding_deployment.simulation.world import FeedingDeploymentPyBulletWorld
from feeding_deployment.simulation.planning import (
    _plan_to_sim_state_trajectory,
    remap_trajectory_to_constant_distance,
    _get_plan_to_execute_grasp,
    _get_plan_to_execute_ungrasp,
)
from feeding_deployment.simulation.control import cartesian_control_step


class FeedingDeploymentPyBulletSimulator(FeedingDeploymentPyBulletWorld):
    """A PyBullet-based simulator for the feeding deployment environment."""

    def __init__(self, scene_description: SceneDescription, use_gui: bool = True, ignore_user = False) -> None:
        
        super().__init__(scene_description, use_gui, ignore_user)
        self.recorded_states: list[FeedingDeploymentWorldState] = []
    
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

    def set_wrist_state(self, pitch_angle, roll_angle):
        self.set_utensil_motors([pitch_angle, roll_angle])

    def plan_to_ee_pose(self, pose: Pose, max_control_time: float = 30.0) -> list[FeedingDeploymentWorldState]:
        """Move the robot to the specified end effector pose using cartesian control."""

        visualize_pose(pose, self.physics_client_id)
        visualize_pose(self.robot.get_end_effector_pose(), self.physics_client_id)
        initial_fingers_positions = self.robot.get_joint_positions()[7:]
    
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
            target_positions = np.concatenate((target_positions, initial_fingers_positions)) # Rajat ToDo: Remove hardcoding
            self.set_robot_motors(target_positions)
        
        if not target_reached:
            raise RuntimeError("Sim cartesian controller: Failed to reach target pose in time")

        plan = _plan_to_sim_state_trajectory(joint_trajectory, self)
        plan = remap_trajectory_to_constant_distance(plan, self)
        
        self.recorded_states.extend(plan)
        return plan

    def plan_to_joint_positions(self, joint_positions: list[float], max_control_time: float = 30.0) -> list[FeedingDeploymentWorldState]:
        """Move the robot to the specified joint positions."""
        
        initial_joint_positions = self.robot.get_joint_positions().copy()
        target_joint_positions = add_fingers_to_joint_positions(self.robot, joint_positions)

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
            # Rajat ToDo: check if this is necessary
            # plan = remap_trajectory_to_constant_distance(plan, self)
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
        
        self.recorded_states.extend(plan)
        return plan
    
    def visualize_plan(self, plan: list[FeedingDeploymentWorldState]) -> None:
        """Visualize a plan in PyBullet."""
        for sim_state in plan:
            self.sync(sim_state)
            time.sleep(0.1)

    def grasp_object(self, object_name: str) -> list[FeedingDeploymentWorldState]:
        plan = _get_plan_to_execute_grasp(self, object_name)
        self.recorded_states.extend(plan)

    def ungrasp_object(self) -> list[FeedingDeploymentWorldState]:
        plan = _get_plan_to_execute_ungrasp(self)
        self.recorded_states.extend(plan)

    def close_gripper(self) -> None:
        raise NotImplementedError("TODO")
        self.robot.close_fingers()

    def open_gripper(self) -> None:
        raise NotImplementedError("TODO")
        self.robot.open_fingers()

    def get_current_state(self) -> FeedingDeploymentWorldState:

        return FeedingDeploymentWorldState(
            robot_joints=self.robot.get_joint_positions(),
            drink_pose=get_pose(self.drink_id, self.physics_client_id),
            wipe_pose=get_pose(self.wipe_id, self.physics_client_id),
            utensil_pose=get_pose(self.utensil_id, self.physics_client_id),
            held_object=self.held_object_name,
            held_object_tf=self.held_object_tf,
        )

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

    def get_transform(self, source_frame, target_frame):

        try:
            if target_frame == "camera_color_optical_frame":
                source_to_ee_frame = get_relative_link_pose(self.robot.robot_id, self.robot.link_from_name(source_frame), self.robot.link_from_name("end_effector_link"), self.physics_client_id)
                ee_frame_to_camera_frame = self.scene_description.camera_pose
                return source_to_ee_frame.multiply(ee_frame_to_camera_frame)
            else:
                source_to_target_frame = get_relative_link_pose(self.robot.robot_id, self.robot.link_from_name(source_frame), self.robot.link_from_name(target_frame), self.physics_client_id)
                return source_to_target_frame
        except:
            raise NotImplementedError(f"{source_frame} to {target_frame} transform not implemented for simulation")