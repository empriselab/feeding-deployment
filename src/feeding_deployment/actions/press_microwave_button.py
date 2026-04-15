from typing import Any

from relational_structs import (
    LiftedAtom,
    LiftedOperator,
    Object,
    Variable,
)
from feeding_deployment.actions.base import (
    HighLevelAction,
    microwave_type,
    GripperFree,
    InFrontOf,
    DoorClosed,
    PlateAt,
    FoodHeated,
)


class PressMicrowaveButtonHLA(HighLevelAction):
    """Press the microwave button."""

    def get_name(self) -> str:
        return "PressMicrowaveButton"

    def get_operator(self) -> LiftedOperator:
        microwave = Variable("?microwave", microwave_type)

        return LiftedOperator(
            self.get_name(),
            parameters=[microwave],
            preconditions={
                LiftedAtom(GripperFree, []),
                LiftedAtom(InFrontOf, [microwave]),
                LiftedAtom(DoorClosed, [microwave]),
                LiftedAtom(PlateAt, [microwave]),
            },
            add_effects={
                LiftedAtom(FoodHeated, []),
            },
            delete_effects=set(),
        )

    def get_behavior_tree_filename(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> str:
        del params
        assert len(objects) == 1
        microwave = objects[0]
        assert self.sim.scene_description.scene_label == "vention"
        assert microwave.name == "microwave", (
            "The object parameter for PressMicrowaveButtonHLA should be the microwave"
        )
        return "press_microwave_button.yaml"

    def press_microwave_button(self, speed: str, duration: float) -> None:
        print(f"Pressing microwave button (duration={duration}s) ...")