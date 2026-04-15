from typing import Any

from relational_structs import (
    LiftedAtom,
    LiftedOperator,
    Object,
    Variable,
)
from feeding_deployment.actions.base import (
    HighLevelAction,
    appliance_type,
    GripperFree,
    InFrontOf,
    DoorOpen,
    DoorClosed,
    SafeToNavigate,
)


class CloseDoorHLA(HighLevelAction):
    """Close a door (fridge or microwave)."""

    def get_name(self) -> str:
        return "CloseDoor"

    def get_operator(self) -> LiftedOperator:
        appliance = Variable("?appliance", appliance_type)
        return LiftedOperator(
            self.get_name(),
            parameters=[appliance],
            preconditions={
                LiftedAtom(GripperFree, []),
                LiftedAtom(InFrontOf, [appliance]),
                LiftedAtom(DoorOpen, [appliance]),
            },
            add_effects={
                LiftedAtom(DoorClosed, [appliance]),
                LiftedAtom(SafeToNavigate, []),
            },
            delete_effects={LiftedAtom(DoorOpen, [appliance])},
        )

    def get_behavior_tree_filename(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> str:
        del params
        assert len(objects) == 1
        appliance = objects[0]
        assert self.sim.scene_description.scene_label == "vention"
        assert appliance.name in ["fridge", "microwave"]
        return f"close_{appliance.name}.yaml"

    def close_fridge(self, speed: str) -> None:
        print("Closing fridge door ...")

    def close_microwave(self, speed: str) -> None:
        print("Closing microwave door ...")