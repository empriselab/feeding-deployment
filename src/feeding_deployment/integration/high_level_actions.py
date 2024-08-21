"""High-level actions that we can simulate and execute."""

import abc

from relational_structs import LiftedOperator, Predicate, Type, Object, LiftedAtom, GroundOperator
from feeding_deployment.robot_controller.arm_client import KinovaCommand

# Define some predicates that can be used for sequencing the high-level actions.
tool_type = Type("tool")  # utensil, cup, or wiping tool
GripperFree = Predicate("GripperFree", [])  # not holding any tool
Holding = Predicate("Holding", [tool_type])  # holding tool
ToolTransferDone = Predicate("ToolTransferDone", [tool_type])  # wiped, drank, or ate
ToolPrepared = Predicate("ToolPrepared", [tool_type])  # e.g., bite acquired


# Define high-level actions.
class HighLevelAction(abc.ABC):
    """Base class for high-level action."""

    def __init__(self, types: tuple[Type]):
        self._types = types

    @abc.abstractmethod
    def get_operator(self) -> LiftedOperator:
        """Create a planning operator for this HLA."""

    @abc.abstractmethod
    def get_robot_commands(self, objects: tuple[Object], sim) -> list[KinovaCommand]:
        """Return a list of commands to send to the robot to execute this.
        
        TODO: add type for sim.
        """

    @abc.abstractmethod
    def update_simulator(self, objects: tuple[Object], sim) -> None:
        """Update the given simulator assuming that the action was executed.
        
        TODO: add typing for sim
        TODO; add argument for any sensory inputs
        """


class PickToolHLA(HighLevelAction):
    """Pick up a tool (utensil, drink, or wipe)."""

    def __init__(self):
        super().__init__((tool_type, ))

    def get_operator(self) -> LiftedOperator:
        tool = tool_type("?tool")
        return LiftedOperator("PickTool",
                                        parameters=[tool],
                                        preconditions={LiftedAtom(GripperFree, [])},
                                        add_effects={Holding([tool])},
                                        delete_effects={LiftedAtom(GripperFree, [])})
    
    def get_robot_commands(self, objects: tuple[Object], sim) -> list[KinovaCommand]:
        import ipdb; ipdb.set_trace()

    def update_simulator(self, objects: tuple[Object], sim) -> None:
        import ipdb; ipdb.set_trace()


    
class StowToolHLA(HighLevelAction):
    """Stow a tool (utensil, drink, or wipe)."""

    def __init__(self):
        super().__init__((tool_type, ))

    def get_operator(self) -> LiftedOperator:
        tool = tool_type("?tool")
        return LiftedOperator("StowTool",
                                parameters=[tool],
                                preconditions={Holding([tool])},
                                add_effects={LiftedAtom(GripperFree, [])},
                                delete_effects={Holding([tool])})
    
    def get_robot_commands(self, objects: tuple[Object], sim) -> list[KinovaCommand]:
        import ipdb; ipdb.set_trace()

    def update_simulator(self, objects: tuple[Object], sim) -> None:
        import ipdb; ipdb.set_trace()


class TransferToolHLA(HighLevelAction):
    """Wipe, or transfer drink, or transfer bite."""
    
    def __init__(self):
        super().__init__((tool_type, ))

    def get_operator(self) -> LiftedOperator:
        tool = tool_type("?tool")
        return LiftedOperator("TransferTool",
                                parameters=[tool],
                                preconditions={Holding([tool]), ToolPrepared([tool])},
                                add_effects={LiftedAtom(ToolTransferDone, [tool])},
                                delete_effects=set())

    def get_robot_commands(self, objects: tuple[Object], sim) -> list[KinovaCommand]:
        import ipdb; ipdb.set_trace()

    def update_simulator(self, objects: tuple[Object], sim) -> None:
        import ipdb; ipdb.set_trace()


class PrepareToolHLA(HighLevelAction):
    """Bite acquisition; other tools are always prepared."""
    
    def __init__(self):
        super().__init__((tool_type, ))

    def get_operator(self) -> LiftedOperator:
        tool = tool_type("?tool")
        return LiftedOperator("PrepareTool",
                            parameters=[tool],
                            preconditions={Holding([tool])},
                            add_effects={ToolPrepared([tool])},
                            delete_effects=set())

    def get_robot_commands(self, objects: tuple[Object], sim) -> list[KinovaCommand]:
        import ipdb; ipdb.set_trace()

    def update_simulator(self, objects: tuple[Object], sim) -> None:
        import ipdb; ipdb.set_trace()


def pddl_plan_to_hla_plan(pddl_plan: list[GroundOperator], hlas: set[HighLevelAction]) -> list[tuple[HighLevelAction, tuple[Object]]]:
    """Convert a PDDL plan into a sequence of HLA and objects."""
    hla_plan = []
    op_to_hla = {hla.get_operator(): hla for hla in hlas}
    for ground_operator in pddl_plan:
        hla = op_to_hla[ground_operator.parent]
        hla_plan.append((hla, ground_operator.parameters))
    return hla_plan
