from typing import Any

from relational_structs import (
    LiftedAtom,
    LiftedOperator,
    Object,
    Variable,
)
from feeding_deployment.actions.base import (
    HighLevelAction,
    nav_target_type,
    InFrontOf,
    DoorClosed,
    SafeToNavigate,
    GripperFree,
)


class NavigateHLA(HighLevelAction):
    """Navigate from one target to another."""

    def get_name(self) -> str:
        return "Navigate"

    def get_operator(self) -> LiftedOperator:
        src = Variable("?from", nav_target_type)
        dst = Variable("?to", nav_target_type)

        return LiftedOperator(
            self.get_name(),
            parameters=[src, dst],
            preconditions={
                LiftedAtom(InFrontOf, [src]),
                LiftedAtom(SafeToNavigate, []),
                LiftedAtom(GripperFree, []),
            },
            add_effects={
                LiftedAtom(InFrontOf, [dst]),
            },
            delete_effects={
                LiftedAtom(InFrontOf, [src]),
            },
        )

    def get_behavior_tree_filename(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> str:
        del params
        assert len(objects) == 2
        _, dst = objects
        assert self.sim.scene_description.scene_label == "vention"
        assert dst.name in ["fridge", "microwave", "sink", "table"]
        return f"navigate_to_{dst.name}.yaml"

    def navigate_to_fridge(self, speed: str) -> None:
        print("Navigating to fridge ...")

    def navigate_to_microwave(self, speed: str) -> None:
        print("Navigating to microwave ...")

    def navigate_to_sink(self, speed: str) -> None:
        print("Navigating to sink ...")

    def navigate_to_table(self, speed: str) -> None:
        print("Navigating to table ...")