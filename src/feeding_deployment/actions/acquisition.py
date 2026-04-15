from typing import Any

import time
import pickle
import numpy as np
import cv2

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
    IsUtensil,
    PlateInView,
    ToolPrepared,
    FoodHeated,
    InFrontOf,
    PlateAt,
)

from feeding_deployment.actions.flair.food_manipulation_skill_library import FoodManipulationSkillLibrary

class AcquireBiteHLA(HighLevelAction):
    """Bite acquisition; other tools are always prepared."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.food_manipulation_skill_library = FoodManipulationSkillLibrary(
            self.sim,
            self.robot_interface,
            self.wrist_interface,
            self.perception_interface,
            self.rviz_interface,
            self.no_waits,
        )
        self.params = None

        self.food_detection_log_dir = self.log_dir / "food_detection_log"
        self.food_detection_log_dir.mkdir(exist_ok=True)

    def get_name(self) -> str:
        return "AcquireBiteWithTool"

    def get_operator(self) -> LiftedOperator:
        tool = Variable("?tool", tool_type)
        table = Variable("?table", table_type)
        return LiftedOperator(
            self.get_name(),
            parameters=[tool, table],
            preconditions={
                LiftedAtom(Holding, [tool]),
                LiftedAtom(IsUtensil, [tool]),
                LiftedAtom(FoodHeated, []),
                LiftedAtom(InFrontOf, [table]),
                LiftedAtom(PlateAt, [table]),
            },
            add_effects={
                LiftedAtom(ToolPrepared, [tool]),
            },
            delete_effects=set(),
        )

    def get_behavior_tree_filename(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> str:
        del params
        assert len(objects) == 2
        tool = objects[0]
        table = objects[1]
        assert tool.name == "utensil"
        assert table.name == "table"
        return "acquire_bite.yaml"
    
    def acquire_bite(self, speed: str, dipping_depth: float, skewering_depth: float, skewering_orientation: str, autocontinue_timeout: float, ask_confirmation: bool) -> None:
        print("Acquiring bite with utensil ...")