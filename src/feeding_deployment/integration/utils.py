"""Utilities for integration."""

import numpy as np

from feeding_deployment.drinking.utils import CupManipulationTrajectory
from feeding_deployment.robot_controller.arm_client import (
    CloseGripperCommand,
    JointTrajectoryCommand,
    KinovaCommand,
    OpenGripperCommand,
)


def cup_manipulation_trajectory_to_kinova_commands(
    traj: CupManipulationTrajectory,
) -> list[KinovaCommand]:
    """The Kinova controller expects arm joints and gripper values."""
    cmds = []
    last_gripper: str | None = None
    current_trajectory = []
    for joint_state in traj.joint_states:
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
