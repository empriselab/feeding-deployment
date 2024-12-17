"""Defines the state of a feeding deployment simulated environment."""

from dataclasses import dataclass

from pybullet_helpers.geometry import Pose
from pybullet_helpers.joint import JointPositions


@dataclass(frozen=True)
class FeedingDeploymentWorldState:
    """The state of a feeding deployment simulated environment."""

    robot_joints: JointPositions
    drink_pose: Pose | None = None  # None if held
    wipe_pose: Pose | None = None  # None if held
    utensil_pose: Pose | None = None  # None if held
    held_object: str | None = None
    held_object_tf: Pose | None = None

    def __post_init__(self) -> None:
        assert self.drink_pose is not None or self.held_object == "drink"
        assert self.wipe_pose is not None or self.held_object == "wipe"
        assert self.utensil_pose is not None or self.held_object == "utensil"
