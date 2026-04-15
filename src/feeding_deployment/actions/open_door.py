from typing import Any

import time

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


class OpenDoorHLA(HighLevelAction):
    """Open a door (fridge or microwave)."""

    def get_name(self) -> str:
        return "OpenDoor"

    def get_operator(self) -> LiftedOperator:
        appliance = Variable("?appliance", appliance_type)
        return LiftedOperator(
            self.get_name(),
            parameters=[appliance],
            preconditions={
                LiftedAtom(GripperFree, []),
                LiftedAtom(InFrontOf, [appliance]),
                LiftedAtom(DoorClosed, [appliance]),
            },
            add_effects={LiftedAtom(DoorOpen, [appliance])},
            delete_effects={
                LiftedAtom(DoorClosed, [appliance]),
                LiftedAtom(SafeToNavigate, []),
            },
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
        return f"open_{appliance.name}.yaml"

    def open_fridge(self, speed: str) -> None:
        print("Opening fridge door ...")
        
    def open_microwave(self, speed: str) -> None:
        print("Opening microwave door ...")