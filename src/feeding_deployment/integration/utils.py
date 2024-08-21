"""Utilities for integration."""

import numpy as np
from numpy.typing import NDArray

from feeding_deployment.robot_controller.arm_client import (
    CloseGripperCommand,
    JointTrajectoryCommand,
    KinovaCommand,
    OpenGripperCommand,
)
from feeding_deployment.simulation.state import FeedingDeploymentSimulatorState


def simulated_trajectory_to_kinova_commands(
    traj: list[FeedingDeploymentSimulatorState],
) -> list[KinovaCommand]:
    """The Kinova controller expects arm joints and gripper values."""
    cmds: list[KinovaCommand] = []
    last_gripper: str | None = None
    current_trajectory: list[NDArray] = []
    for state in traj:
        joint_state = state.robot_joints
        assert len(joint_state) == 9  # making assumptions about Kinova
        arm = np.array(joint_state[:7])
        assert np.isclose(joint_state[7], joint_state[8])
        gripper = "closed" if joint_state[8] == 0 else "open"
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
