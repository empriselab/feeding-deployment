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
    GripperFree,
    Holding,
    table_type,
    InFrontOf,
)

class StowToolHLA(HighLevelAction):
    """Stow a tool (utensil, drink, or wipe)."""

    def get_name(self) -> str:
        return "StowTool"

    def get_operator(self) -> LiftedOperator:
        tool = Variable("?tool", tool_type)
        table = Variable("?table", table_type)
        return LiftedOperator(
            self.get_name(),
            parameters=[tool, table],
            preconditions={
                LiftedAtom(Holding, [tool]),
                LiftedAtom(InFrontOf, [table])
            },
            add_effects={LiftedAtom(GripperFree, [])},
            delete_effects={LiftedAtom(Holding, [tool])},
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
        assert tool.name in ["utensil", "drink", "wipe"]
        assert table.name == "table"
        return f"stow_{tool.name}.yaml"

    def stow_utensil(self, speed: str) -> None:
        print("Stowing utensil ...")

    def stow_drink(self, speed: str) -> None:
        print("Stowing drink ...")

    def stow_wipe(self, speed: str) -> None:
        print("Stowing wipe ...")

    def stow_plate(self, speed: str) -> None:
        print("Stowing plate ...")
