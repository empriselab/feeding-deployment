from typing import Any

from relational_structs import (
    LiftedAtom,
    LiftedOperator,
    Object,
    Variable,
)
from feeding_deployment.actions.base import (
    HighLevelAction,
    tool_type,
    plate_type,
    table_type,
    sink_type,
    holder_type,
    appliance_type,
    GripperFree,
    Holding,
    InFrontOf,
    DoorOpen,
    PlateAt,
)

class PlacePlateInApplianceHLA(HighLevelAction):
    """Place the plate into an appliance (fridge or microwave)."""

    def get_name(self) -> str:
        return "PlacePlateInAppliance"

    def get_operator(self) -> LiftedOperator:
        plate = Variable("?plate", plate_type)
        appliance = Variable("?appliance", appliance_type)

        return LiftedOperator(
            self.get_name(),
            parameters=[plate, appliance],
            preconditions={
                LiftedAtom(Holding, [plate]),
                LiftedAtom(InFrontOf, [appliance]),
                LiftedAtom(DoorOpen, [appliance]),
            },
            add_effects={
                LiftedAtom(GripperFree, []),
                LiftedAtom(PlateAt, [appliance]),
            },
            delete_effects={
                LiftedAtom(Holding, [plate]),
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
        assert plate.name == "plate"
        assert appliance.name in ["fridge", "microwave"]

        return f"place_plate_in_{appliance.name}.yaml"

    def place_plate_in_fridge(self, speed: str) -> None:
        print("Placing plate in fridge ...")

    def place_plate_in_microwave(self, speed: str) -> None:
        print("Placing plate in microwave ...")


class PlacePlateOnHolderHLA(HighLevelAction):
    """Place the plate onto the holder."""

    def get_name(self) -> str:
        return "PlacePlateOnHolder"

    def get_operator(self) -> LiftedOperator:
        plate = Variable("?plate", plate_type)
        holder = Variable("?holder", holder_type)

        return LiftedOperator(
            self.get_name(),
            parameters=[plate, holder],
            preconditions={
                LiftedAtom(Holding, [plate]),
            },
            add_effects={
                LiftedAtom(GripperFree, []),
                LiftedAtom(PlateAt, [holder]),
            },
            delete_effects={
                LiftedAtom(Holding, [plate]),
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
        return "place_plate_on_holder.yaml"

    def place_plate_on_holder(self, speed: str) -> None:
        print("Placing plate on holder ...")

class PlacePlateInSinkHLA(HighLevelAction):
    """Place the plate in the sink."""

    def get_name(self) -> str:
        return "PlacePlateInSink"
    
    def get_operator(self) -> LiftedOperator:
        plate = Variable("?plate", plate_type)
        sink = Variable("?sink", sink_type)

        return LiftedOperator(
            self.get_name(),
            parameters=[plate, sink],
            preconditions={
                LiftedAtom(Holding, [plate]),
                LiftedAtom(InFrontOf, [sink]),
            },
            add_effects={
                LiftedAtom(GripperFree, []),
                LiftedAtom(PlateAt, [sink]),
            },
            delete_effects={
                LiftedAtom(Holding, [plate]),
            },
        )

    def get_behavior_tree_filename(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> str:
        del params
        assert len(objects) == 2
        plate, sink = objects
        assert self.sim.scene_description.scene_label == "vention"
        assert plate.name == "plate"
        assert sink.name == "sink"

        return f"place_plate_in_sink.yaml"

    def place_plate_in_sink(self, speed: str) -> None:
        print("Placing plate in sink ...")

class PlacePlateOnTableHLA(HighLevelAction):
    """Place the plate on the table."""

    def get_name(self) -> str:
        return "PlacePlateOnTable"
    
    def get_operator(self) -> LiftedOperator:
        plate = Variable("?plate", plate_type)
        table = Variable("?table", table_type)
        
        return LiftedOperator(
            self.get_name(),
            parameters=[plate, table],
            preconditions={
                LiftedAtom(Holding, [plate]),
                LiftedAtom(InFrontOf, [table]),
            },
            add_effects={
                LiftedAtom(GripperFree, []),
                LiftedAtom(PlateAt, [table]),
            },
            delete_effects={
                LiftedAtom(Holding, [plate]),
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

        return f"place_plate_on_table.yaml"

    def place_plate_on_table(self, speed: str) -> None:
        print("Placing plate on table ...")