"""High-level actions that we can simulate and execute."""

import abc

from relational_structs import (
    GroundOperator,
    LiftedAtom,
    LiftedOperator,
    Object,
    Predicate,
    Type,
)

from feeding_deployment.integration.utils import simulated_trajectory_to_kinova_commands
from feeding_deployment.robot_controller.arm_client import Arm, KinovaCommand
from feeding_deployment.simulation.planning import (
    get_plan_to_grasp_cup,
    remap_trajectory_to_constant_distance,
)
from feeding_deployment.simulation.simulator import FeedingDeploymentPyBulletSimulator
from feeding_deployment.simulation.state import FeedingDeploymentSimulatorState

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
        perception_interface,
    ) -> None:
        self._sim = sim
        self._robot_interface = robot_interface
        self._perception_interface = perception_interface

    @abc.abstractmethod
    def get_operator(self) -> LiftedOperator:
        """Create a planning operator for this HLA."""

    @abc.abstractmethod
    def get_simulated_trajectory(
        self,
        objects: tuple[Object],
    ) -> list[FeedingDeploymentSimulatorState]:
        """Update the given simulator assuming that the action was executed."""

    def get_robot_commands(
        self,
        objects: tuple[Object],
        sim_traj: list[FeedingDeploymentSimulatorState],
    ) -> list[KinovaCommand]:
        """Default implementation follows sim_traj exactly, but subclasses can
        override to modify their execution."""
        del objects  # not used
        return simulated_trajectory_to_kinova_commands(sim_traj)


class PickToolHLA(HighLevelAction):
    """Pick up a tool (utensil, drink, or wipe)."""

    def get_operator(self) -> LiftedOperator:
        tool = tool_type("?tool")
        return LiftedOperator(
            "PickTool",
            parameters=[tool],
            preconditions={LiftedAtom(GripperFree, [])},
            add_effects={Holding([tool])},
            delete_effects={LiftedAtom(GripperFree, [])},
        )

    def get_simulated_trajectory(
        self,
        objects: tuple[Object],
    ) -> list[FeedingDeploymentSimulatorState]:
        assert len(objects) == 1
        tool = objects[0]
        if tool.name == "cup":
            nominal_plan = get_plan_to_grasp_cup(self._sim)
            remapped_plan = remap_trajectory_to_constant_distance(
                nominal_plan, self._sim
            )
            return remapped_plan
        # TODO
        print(f"PickTool not yet implemented for {tool}")
        return []


class StowToolHLA(HighLevelAction):
    """Stow a tool (utensil, drink, or wipe)."""

    def get_operator(self) -> LiftedOperator:
        tool = tool_type("?tool")
        return LiftedOperator(
            "StowTool",
            parameters=[tool],
            preconditions={Holding([tool])},
            add_effects={LiftedAtom(GripperFree, [])},
            delete_effects={Holding([tool])},
        )

    def get_simulated_trajectory(
        self,
        objects: tuple[Object],
    ) -> list[FeedingDeploymentSimulatorState]:
        # TODO
        assert len(objects) == 1
        tool = objects[0]
        print(f"StowTool not yet implemented for {tool}")
        return []


class TransferToolHLA(HighLevelAction):
    """Wipe, or transfer drink, or transfer bite."""

    def get_operator(self) -> LiftedOperator:
        tool = tool_type("?tool")
        return LiftedOperator(
            "TransferTool",
            parameters=[tool],
            preconditions={Holding([tool]), ToolPrepared([tool])},
            add_effects={LiftedAtom(ToolTransferDone, [tool])},
            delete_effects=set(),
        )

    def get_simulated_trajectory(
        self,
        objects: tuple[Object],
    ) -> list[FeedingDeploymentSimulatorState]:
        # TODO
        assert len(objects) == 1
        tool = objects[0]
        print(f"TransferTool not yet implemented for {tool}")
        return []


class PrepareToolHLA(HighLevelAction):
    """Bite acquisition; other tools are always prepared."""

    def get_operator(self) -> LiftedOperator:
        tool = tool_type("?tool")
        return LiftedOperator(
            "PrepareTool",
            parameters=[tool],
            preconditions={Holding([tool])},
            add_effects={ToolPrepared([tool])},
            delete_effects=set(),
        )

    def get_simulated_trajectory(
        self,
        objects: tuple[Object],
    ) -> list[FeedingDeploymentSimulatorState]:
        # TODO
        assert len(objects) == 1
        tool = objects[0]
        print(f"PrepareTool not yet implemented for {tool}")
        return []


def pddl_plan_to_hla_plan(
    pddl_plan: list[GroundOperator], hlas: set[HighLevelAction]
) -> list[tuple[HighLevelAction, tuple[Object]]]:
    """Convert a PDDL plan into a sequence of HLA and objects."""
    hla_plan = []
    op_to_hla = {hla.get_operator(): hla for hla in hlas}
    for ground_operator in pddl_plan:
        hla = op_to_hla[ground_operator.parent]
        hla_plan.append((hla, ground_operator.parameters))
    return hla_plan
