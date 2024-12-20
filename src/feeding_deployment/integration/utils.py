"""Utilities for integration."""

import numpy as np
from numpy.typing import NDArray

from feeding_deployment.robot_controller.command_interface import (
    CloseGripperCommand,
    JointTrajectoryCommand,
    KinovaCommand,
    OpenGripperCommand,
)
from feeding_deployment.simulation.state import FeedingDeploymentWorldState


def simulated_trajectory_to_kinova_commands(
    traj: list[FeedingDeploymentWorldState],
) -> list[KinovaCommand]:
    """The Kinova controller expects arm joints and gripper values."""
    cmds: list[KinovaCommand] = []
    last_gripper: str | None = None
    current_trajectory: list[NDArray] = []
    for state in traj:
        joint_state = state.robot_joints
        assert len(joint_state) == 13  # making assumptions about Kinova
        arm = np.array(joint_state[:7])
        gripper = "closed" if joint_state[7] == 0 else "open"
        if last_gripper is None:
            last_gripper = gripper
        elif gripper != last_gripper:
            if current_trajectory:
                cmds.append(JointTrajectoryCommand(current_trajectory))
                current_trajectory = []
            if gripper == "closed":
                cmds.append(CloseGripperCommand())
            else:
                assert gripper == "open"
                cmds.append(OpenGripperCommand())
        current_trajectory.append(arm)
        last_gripper = gripper
    if current_trajectory:
        cmds.append(JointTrajectoryCommand(current_trajectory))
    return cmds
