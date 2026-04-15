from typing import Any

import time

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
from feeding_deployment.actions.base import (
    HighLevelAction,
    tool_type,
    table_type,
    GripperFree,
    Holding,
    InFrontOf,
)

class PickToolHLA(HighLevelAction):
    """Pick up a tool (utensil, drink, or wipe)."""

    def get_name(self) -> str:
        return "PickTool"

    def get_operator(self) -> LiftedOperator:
        tool = Variable("?tool", tool_type)
        table = Variable("?table", table_type)
        return LiftedOperator(
            self.get_name(),
            parameters=[tool, table],
            preconditions={
                LiftedAtom(GripperFree, []),
                LiftedAtom(InFrontOf, [table]),
            },
            add_effects={LiftedAtom(Holding, [tool])},
            delete_effects={LiftedAtom(GripperFree, [])},
        )
    
    def get_behavior_tree_filename(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> str:
        del params  # not used right now
        assert len(objects) == 2
        tool = objects[0]
        table = objects[1]
        assert self.sim.scene_description.scene_label == "vention"
        assert tool.name in ["utensil", "drink", "wipe"]
        assert table.name == "table"
        return f"pick_{tool.name}.yaml"
    
    def pick_utensil(self, speed: str) -> None:
        print("Picking utensil ...")
        
    def pick_drink(self, speed: str) -> None:
        print("Picking drinking ...")

    def pick_wipe(self, speed: str) -> None:
        print("Picking wipe ...")

    def pick_plate(self, speed: str) -> None:
        print("Picking plate ...")
