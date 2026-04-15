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
    plate_type,
    table_type,
    appliance_type,
    holder_type,
    GripperFree,
    Holding,
    InFrontOf,
    DoorOpen,
    PlateAt,
)


class PickPlateFromApplianceHLA(HighLevelAction):
    """Pick the plate from an appliance (fridge or microwave)."""

    def get_name(self) -> str:
        return "PickPlateFromAppliance"

    def get_operator(self) -> LiftedOperator:
        plate = Variable("?plate", plate_type)
        appliance = Variable("?appliance", appliance_type)

        return LiftedOperator(
            self.get_name(),
            parameters=[plate, appliance],
            preconditions={
                LiftedAtom(GripperFree, []),
                LiftedAtom(InFrontOf, [appliance]),
                LiftedAtom(DoorOpen, [appliance]),
                LiftedAtom(PlateAt, [appliance]),
            },
            add_effects={
                LiftedAtom(Holding, [plate]),
            },
            delete_effects={
                LiftedAtom(GripperFree, []),
                LiftedAtom(PlateAt, [appliance]),
            },
        )

    def get_behavior_tree_filename(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> str:
        del params
        assert len(objects) == 2
        plate, appliance = objects
        assert self.sim.scene_description.scene_label == "vention"
        print("Plate name:", plate.name)
        assert plate.name == "plate"
        assert appliance.name in ["fridge", "microwave"]

        return f"pick_plate_from_{appliance.name}.yaml"
    
    def pick_plate_from_fridge(self, speed: str) -> None:
        print("Picking plate from fridge ...")

    def pick_plate_from_microwave(self, speed: str) -> None:
        print("Picking plate from microwave ...")


class PickPlateFromHolderHLA(HighLevelAction):
    """Pick the plate from the holder."""

    def get_name(self) -> str:
        return "PickPlateFromHolder"

    def get_operator(self) -> LiftedOperator:
        plate = Variable("?plate", plate_type)
        holder = Variable("?holder", holder_type)

        return LiftedOperator(
            self.get_name(),
            parameters=[plate, holder],
            preconditions={
                LiftedAtom(GripperFree, []),
                LiftedAtom(PlateAt, [holder]),
            },
            add_effects={
                LiftedAtom(Holding, [plate]),
            },
            delete_effects={
                LiftedAtom(GripperFree, []),
                LiftedAtom(PlateAt, [holder]),
            },
        )

    def get_behavior_tree_filename(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> str:
        del params
        assert len(objects) == 2
        plate, holder = objects
        assert self.sim.scene_description.scene_label == "vention"
        assert plate.name == "plate"
        assert holder.name == "holder"
        return "pick_plate_from_holder.yaml"
    
    def pick_plate_from_holder(self, speed: str) -> None:
        print("Picking plate from holder ...")

class PickPlateFromTableHLA(HighLevelAction):
    """Pick the plate from the table."""

    def get_name(self) -> str:
        return "PickPlateFromTable"

    def get_operator(self) -> LiftedOperator:
        plate = Variable("?plate", plate_type)
        table = Variable("?table", table_type)

        return LiftedOperator(
            self.get_name(),
            parameters=[plate, table],
            preconditions={
                LiftedAtom(GripperFree, []),
                LiftedAtom(InFrontOf, [table]),
                LiftedAtom(PlateAt, [table]),
            },
            add_effects={
                LiftedAtom(Holding, [plate]),
            },
            delete_effects={
                LiftedAtom(GripperFree, []),
                LiftedAtom(PlateAt, [table]),
            },
        )

    def get_behavior_tree_filename(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> str:
        del params
        assert len(objects) == 2
        plate, table = objects
        assert self.sim.scene_description.scene_label == "vention"
        assert plate.name == "plate"
        assert table.name == "table"
        return "pick_plate_from_table.yaml"

    def pick_plate_from_table(self, speed: str) -> None:
        print("Picking plate from table ...")