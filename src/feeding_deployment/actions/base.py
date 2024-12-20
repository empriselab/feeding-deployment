"""High-level actions that we can simulate and execute."""

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import time

from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.joint import JointPositions
from relational_structs import (
    GroundAtom,
    GroundOperator,
    LiftedAtom,
    LiftedOperator,
    Object,
    Predicate,
    Type,
    Variable,
)
from pybullet_helpers.gui import visualize_pose

from feeding_deployment.interfaces.perception_interface import PerceptionInterface
from feeding_deployment.interfaces.web_interface import WebInterface
from feeding_deployment.interfaces.rviz_interface import RVizInterface
from feeding_deployment.robot_controller.arm_client import ArmInterfaceClient
from feeding_deployment.wrist_controller.wrist_controller import WristInterface
from feeding_deployment.robot_controller.command_interface import (
    CartesianCommand,
    CloseGripperCommand,
    JointCommand,
    KinovaCommand,
    OpenGripperCommand,
)

from feeding_deployment.simulation.planning import (
    _get_plan_to_execute_grasp,
    _get_plan_to_execute_ungrasp,
)
from feeding_deployment.simulation.simulator import FeedingDeploymentPyBulletSimulator
from feeding_deployment.simulation.state import FeedingDeploymentWorldState

# Define some predicates that can be used for sequencing the high-level actions.
tool_type = Type("tool")  # utensil, drink, or wiping tool
GripperFree = Predicate("GripperFree", [])  # not holding any tool
Holding = Predicate("Holding", [tool_type])  # holding tool
ToolTransferDone = Predicate("ToolTransferDone", [tool_type])  # wiped, drank, or ate
ToolPrepared = Predicate("ToolPrepared", [tool_type])  # e.g., bite acquired
PlateInView = Predicate("PlateInView", [])  # of the hand camera
ResetPos = Predicate("ResetPos", [])  # robot in reset position
IsUtensil = Predicate("IsUtensil", [tool_type])

# Define high-level actions.
class HighLevelAction(abc.ABC):
    """Base class for high-level action."""

    def __init__(
        self,
        sim: FeedingDeploymentPyBulletSimulator,
        robot_interface: ArmInterfaceClient,
        perception_interface: PerceptionInterface,
        rviz_interface: RVizInterface,
        web_interface: WebInterface,
        hla_hyperparams: dict[str, Any],
        wrist_interface: WristInterface,
        flair,
        no_waits=False,
        log_path=None
    ) -> None:
        self.sim = sim
        self.robot_interface = robot_interface
        self.perception_interface = perception_interface
        self.rviz_interface = rviz_interface
        self.web_interface = web_interface
        self.hla_hyperparams = hla_hyperparams
        self.wrist_interface = wrist_interface
        self.flair = flair
        self.no_waits = no_waits
        self.log_path = log_path

    @abc.abstractmethod
    def get_name(self) -> str:
        """Get a human-readable name for this HLA."""

    @abc.abstractmethod
    def get_operator(self) -> LiftedOperator:
        """Create a planning operator for this HLA."""

    @abc.abstractmethod
    def execute_action(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> None:
        """Execute the action on the robot and return simulated trajectory."""

    def move_to_joint_positions(self, joint_positions: list[float]) -> None:
        plan = self.sim.plan_to_joint_positions(joint_positions)
        print("Plan has length", len(plan))
        if self.robot_interface is None:
            self.sim.visualize_plan(plan)
        else:
            self.execute_robot_command(JointCommand(pos=joint_positions), plan)
            
    def move_to_ee_pose(self, pose: Pose) -> None:
        plan = self.sim.plan_to_ee_pose(pose)
        if self.robot_interface is None:
            self.sim.visualize_plan(plan)
        else:
            self.execute_robot_command(CartesianCommand(pos=pose.position, quat=pose.orientation), plan)
    
    def grasp_tool(self, tool: str) -> None:
        self.sim.grasp_object(tool)
        if self.robot_interface is not None:
            self.execute_robot_command(OpenGripperCommand(), tool_update=tool)

    def ungrasp_tool(self, tool: str) -> None:
        self.sim.ungrasp_object()
        if self.robot_interface is not None:
            self.execute_robot_command(CloseGripperCommand(), tool_update=tool)

    def open_gripper(self) -> None:
        if self.robot_interface is None:
            self.sim.robot.open_fingers()
        else:
            self.execute_robot_command(OpenGripperCommand())
    
    def close_gripper(self) -> None:
        if self.robot_interface is None:
            self.sim.robot.close_fingers()
        else:
            self.execute_robot_command(CloseGripperCommand())

    def execute_robot_command(self, robot_command: KinovaCommand, plan_viz: list[FeedingDeploymentWorldState] = None, tool_update: str = None) -> None:
        """Execute the given commands on the robot."""
        if self.robot_interface is None:
            return
        if not self.no_waits:
            if tool_update is not None:
                self.rviz_interface.tool_update(True, tool_update, Pose((0, 0, 0), (0, 0, 0, 1))) # pickup the drink
            if plan_viz is not None:
                self.rviz_interface.visualize_plan(plan_viz)
            input("Execute next command?")
        self.robot_interface.execute_command(robot_command)

@dataclass(frozen=True)
class GroundHighLevelAction:
    """A high-level action with objects and parameters specified.

    For example, the parameters for bite acquisition might include the
    preferred food item. These parameters can be populated automatically
    or set by the user.
    """

    hla: HighLevelAction  # want this to be executed
    objects: tuple[Object, ...]  # for grounding the high-level action
    params: dict = field(default_factory=lambda: {})  # see docstring

    def __str__(self) -> str:
        obj_str = ", ".join([o.name for o in self.objects])
        return f"{self.hla.get_name()}({obj_str})"

    def get_operator(self) -> GroundOperator:
        """Get the operator for this ground HLA."""
        return self.hla.get_operator().ground(self.objects)

    def get_preconditions(self) -> set[GroundAtom]:
        """Get the preconditions for executing this command."""
        return self.get_operator().preconditions

    def execute_action(self) -> None:
        """Execute the command."""
        self.hla.execute_action(self.objects, self.params)

class ResetHLA(HighLevelAction):
    """Move the robot to retract position without any tool."""

    def get_name(self) -> str:
        return "Reset"

    def get_operator(self) -> LiftedOperator:
        return LiftedOperator(
            self.get_name(),
            parameters=[],
            preconditions={LiftedAtom(GripperFree, [])},
            add_effects={LiftedAtom(ResetPos, [])},
            delete_effects=set(),
        )

    def execute_action(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> None:
        assert len(objects) == 0
        assert self.sim.held_object_name is None
        sim_states: list[FeedingDeploymentWorldState] = []
        robot_commands = []

        self.move_to_joint_positions(self.sim.scene_description.retract_pos)

        # set FLAIR preferences to None
        if self.flair is not None:
            self.flair.clear_preference()

def pddl_plan_to_hla_plan(
    pddl_plan: list[GroundOperator], hlas: set[HighLevelAction]
) -> list[GroundHighLevelAction]:
    """Convert a PDDL plan into a sequence of HLA and objects."""
    hla_plan = []
    op_to_hla = {hla.get_operator(): hla for hla in hlas}
    for ground_operator in pddl_plan:
        hla = op_to_hla[ground_operator.parent]
        objects = tuple(ground_operator.parameters)
        ground_hla = GroundHighLevelAction(hla, objects)
        hla_plan.append(ground_hla)
    return hla_plan
