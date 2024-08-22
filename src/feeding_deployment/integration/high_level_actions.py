"""High-level actions that we can simulate and execute."""

import abc
from typing import Any
from pathlib import Path

from relational_structs import (
    GroundOperator,
    LiftedAtom,
    LiftedOperator,
    Object,
    Predicate,
    Type,
    Variable,
)

from feeding_deployment.integration.perception_interface import PerceptionInterface
from feeding_deployment.integration.utils import simulated_trajectory_to_kinova_commands
from feeding_deployment.robot_controller.arm_client import Arm, KinovaCommand
from feeding_deployment.simulation.planning import (
    get_bite_transfer_plan,
    get_plan_to_grasp_cup,
    remap_trajectory_to_constant_distance,
)
from feeding_deployment.simulation.simulator import FeedingDeploymentPyBulletSimulator
from feeding_deployment.simulation.state import FeedingDeploymentSimulatorState

from feeding_deployment.simulation.video import make_simulation_video

# Define some predicates that can be used for sequencing the high-level actions.
tool_type = Type("tool")  # utensil, cup, or wiping tool
GripperFree = Predicate("GripperFree", [])  # not holding any tool
Holding = Predicate("Holding", [tool_type])  # holding tool
ToolTransferDone = Predicate("ToolTransferDone", [tool_type])  # wiped, drank, or ate
ToolPrepared = Predicate("ToolPrepared", [tool_type])  # e.g., bite acquired


# Define high-level actions.
class HighLevelAction(abc.ABC):
    """Base class for high-level action."""

    def __init__(
        self,
        sim: FeedingDeploymentPyBulletSimulator,
        robot_interface: Arm,
        perception_interface: PerceptionInterface,
        hla_hyperparams: dict[str, Any],
    ) -> None:
        self._sim = sim
        self._robot_interface = robot_interface
        self._perception_interface = perception_interface
        self._hla_hyperparams = hla_hyperparams

    @abc.abstractmethod
    def get_operator(self) -> LiftedOperator:
        """Create a planning operator for this HLA."""

    @abc.abstractmethod
    def execute_action(self, run_on_robot, make_videos, objects: tuple[Object, ...]) -> None:
        """Plan and execute the action on the robot."""

# Define a high-level action that follows a planning and then execution pipeline.
class PlanExecuteHighLevelAction(HighLevelAction):
    """Base class for high-level actions that follow planning and then execution pipeline"""
    def execute_action(self, run_on_robot, make_videos, objects: tuple[Object, ...]) -> None:
        """Default implementation uses get_simulated_trajectory, get_robot_commands, and execute_robot_commands
        in sequence, but subclasses can override to modify their execution."""
        sim_traj = self.get_simulated_trajectory(objects)

        # Optionally make a video of the simulated trajectory.
        if make_videos:
            outfile = Path(__file__).parent / "last.mp4"
            make_simulation_video(self.sim, sim_traj, outfile)
        
        robot_commands = self.get_robot_commands(objects, sim_traj)
        if run_on_robot:
            self.execute_robot_commands(robot_commands)
        return sim_traj
    
    @abc.abstractmethod
    def get_simulated_trajectory(
        self,
        objects: tuple[Object, ...],
    ) -> list[FeedingDeploymentSimulatorState]:
        """Update the given simulator assuming that the action was executed."""

    def get_robot_commands(
        self,
        objects: tuple[Object, ...],
        sim_traj: list[FeedingDeploymentSimulatorState],
    ) -> list[KinovaCommand]:
        """Default implementation follows sim_traj exactly, but subclasses can
        override to modify their execution."""
        del objects  # not used
        return simulated_trajectory_to_kinova_commands(sim_traj)
    
    def execute_robot_commands(self, robot_commands: list[KinovaCommand]) -> None:
        """Execute the given commands on the robot."""
        for robot_command in robot_commands:
            self._robot_interface.execute_command(robot_command)


class PickToolHLA(PlanExecuteHighLevelAction):
    """Pick up a tool (utensil, drink, or wipe)."""

    def get_operator(self) -> LiftedOperator:
        tool = Variable("?tool", tool_type)
        return LiftedOperator(
            "PickTool",
            parameters=[tool],
            preconditions={LiftedAtom(GripperFree, [])},
            add_effects={Holding([tool])},
            delete_effects={LiftedAtom(GripperFree, [])},
        )

    def get_simulated_trajectory(
        self,
        objects: tuple[Object, ...],
    ) -> list[FeedingDeploymentSimulatorState]:
        assert len(objects) == 1
        tool = objects[0]
        if tool.name == "cup":
            nominal_plan = get_plan_to_grasp_cup(
                self._sim,
                max_motion_plan_time=self._hla_hyperparams["max_motion_planning_time"],
            )
            remapped_plan = remap_trajectory_to_constant_distance(
                nominal_plan, self._sim
            )
            return remapped_plan
        # TODO
        print(f"PickTool not yet implemented for {tool}")
        return []


class StowToolHLA(PlanExecuteHighLevelAction):
    """Stow a tool (utensil, drink, or wipe)."""

    def get_operator(self) -> LiftedOperator:
        tool = Variable("?tool", tool_type)
        return LiftedOperator(
            "StowTool",
            parameters=[tool],
            preconditions={Holding([tool])},
            add_effects={LiftedAtom(GripperFree, [])},
            delete_effects={Holding([tool])},
        )

    def get_simulated_trajectory(
        self,
        objects: tuple[Object, ...],
    ) -> list[FeedingDeploymentSimulatorState]:
        # TODO
        assert len(objects) == 1
        tool = objects[0]
        print(f"StowTool not yet implemented for {tool}")
        return []


class TransferToolHLA(PlanExecuteHighLevelAction):
    """Wipe, or transfer drink, or transfer bite."""

    def get_operator(self) -> LiftedOperator:
        tool = Variable("?tool", tool_type)
        return LiftedOperator(
            "TransferTool",
            parameters=[tool],
            preconditions={Holding([tool]), ToolPrepared([tool])},
            add_effects={LiftedAtom(ToolTransferDone, [tool])},
            delete_effects=set(),
        )

    def get_simulated_trajectory(
        self,
        objects: tuple[Object, ...],
    ) -> list[FeedingDeploymentSimulatorState]:
        # TODO
        assert len(objects) == 1
        tool = objects[0]
        if tool.name == "utensil":
            forque_target_pose = (
                self._perception_interface.get_head_perception_forque_target_pose()
            )
            nominal_plan = get_bite_transfer_plan(
                forque_target_pose,
                self._sim,
                max_motion_plan_time=self._hla_hyperparams["max_motion_planning_time"],
            )
            remapped_plan = remap_trajectory_to_constant_distance(
                nominal_plan, self._sim
            )
            return remapped_plan
        print(f"TransferTool not yet implemented for {tool}")
        return []


class PrepareToolHLA(HighLevelAction):
    """Bite acquisition; other tools are always prepared."""

    def __init__(
        self,
        sim: FeedingDeploymentPyBulletSimulator,
        robot_interface: Arm,
        perception_interface: PerceptionInterface,
        hla_hyperparams: dict[str, Any],
    ) -> None:
        super().__init__(sim, robot_interface, perception_interface, hla_hyperparams)
        
        # Rajat todo: how do I initialize FLAIR just for the utensil tool?

    def get_operator(self) -> LiftedOperator:
        tool = Variable("?tool", tool_type)
        return LiftedOperator(
            "PrepareTool",
            parameters=[tool],
            preconditions={Holding([tool])},
            add_effects={ToolPrepared([tool])},
            delete_effects=set(),
        )
    
    def execute_action(self, run_on_robot, make_videos, objects: tuple[Object, ...]) -> None:
        assert len(objects) == 1
        tool = objects[0]
        if tool.name == "utensil":
            # Do Bite Acquisition
            print("Doing Bite Acquisition")
        else:
            # Other tools are always prepared
            pass

def pddl_plan_to_hla_plan(
    pddl_plan: list[GroundOperator], hlas: set[HighLevelAction]
) -> list[tuple[HighLevelAction, tuple[Object, ...]]]:
    """Convert a PDDL plan into a sequence of HLA and objects."""
    hla_plan = []
    op_to_hla = {hla.get_operator(): hla for hla in hlas}
    for ground_operator in pddl_plan:
        hla = op_to_hla[ground_operator.parent]
        hla_plan.append((hla, tuple(ground_operator.parameters)))
    return hla_plan
