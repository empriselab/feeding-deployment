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
    BehaviorTreeNode,
    load_behavior_tree,
    tool_type,
    GripperFree,
    Holding,
)

class StowToolHLA(HighLevelAction):
    """Stow a tool (utensil, drink, or wipe)."""

    def get_name(self) -> str:
        return "StowTool"

    def get_operator(self) -> LiftedOperator:
        tool = Variable("?tool", tool_type)
        return LiftedOperator(
            self.get_name(),
            parameters=[tool],
            preconditions={Holding([tool])},
            add_effects={LiftedAtom(GripperFree, [])},
            delete_effects={Holding([tool])},
        )
    
    def get_behavior_tree(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> BehaviorTreeNode:
        del params  # not used right now
        
        assert len(objects) == 1
        tool = objects[0]

        if tool.name == "utensil":
            yaml_filename = "stow_utensil.yaml"
        elif tool.name == "drink":
            yaml_filename = "stow_drink.yaml"
        else:
            raise NotImplementedError

        return load_behavior_tree(yaml_filename, self)

    def execute_action(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> None:
        assert len(objects) == 1
        tool = objects[0]

        if tool.name in ["drink", "utensil"]:

            # Get and execute the behavior tree.
            behavior_tree = self.get_behavior_tree(objects, params)
            behavior_tree.tick()
        
        elif tool.name == "wipe":

            assert self.sim.held_object_name == "wipe"
            
            self.move_to_joint_positions(self.sim.scene_description.retract_pos)

            if self.sim.scene_description.scene_label == "vention":
                self.move_to_joint_positions(self.sim.scene_description.wipe_neutral_pos)
                self.move_to_joint_positions(self.sim.scene_description.wipe_outside_mount_pos)
            elif self.sim.scene_description.scene_label == "wheelchair":
                self.move_to_joint_positions(self.sim.scene_description.wipe_outside_above_mount_pos)
                self.move_to_ee_pose(self.sim.scene_description.wipe_outside_mount)
            self.move_to_ee_pose(self.sim.scene_description.wipe_inside_mount)
            self.ungrasp_tool("wipe")
            self.move_to_ee_pose(self.sim.scene_description.wipe_above_mount)

            if self.sim.scene_description.scene_label == "vention":
                self.move_to_ee_pose(self.sim.scene_description.wipe_infront_mount)

            self.move_to_joint_positions(self.sim.scene_description.retract_pos)

        else:
            print(f"StowTool not yet implemented for {tool}")
            return []

    def stow_utensil(self) -> None:
        assert self.sim.held_object_name == "utensil"
        
        self.move_to_joint_positions(self.sim.scene_description.retract_pos)

        if self.sim.scene_description.scene_label == "vention":
            self.move_to_joint_positions(self.sim.scene_description.utensil_outside_above_mount_pos)
            self.move_to_ee_pose(self.sim.scene_description.utensil_outside_mount)
        elif self.sim.scene_description.scene_label == "wheelchair":
            self.move_to_joint_positions(self.sim.scene_description.utensil_outside_mount_pos)

        self.move_to_ee_pose(self.sim.scene_description.utensil_inside_mount)
        self.ungrasp_tool("utensil")
        self.move_to_ee_pose(self.sim.scene_description.utensil_above_mount)
        self.move_to_joint_positions(self.sim.scene_description.retract_pos)

    def stow_drink(self) -> None:
        assert self.sim.held_object_name == "drink"
        
        self.move_to_joint_positions(self.sim.scene_description.before_transfer_pos)

        last_drink_poses, last_drink_pickup_joint_pos = self.perception_interface.get_last_drink_pickup_configs(study_poses=True)

        self.move_to_joint_positions(last_drink_pickup_joint_pos)
        self.move_to_ee_pose(last_drink_poses['inside_top_pose'])
        self.ungrasp_tool("drink")
        self.move_to_ee_pose(last_drink_poses['place_inside_bottom_pose'])
        self.move_to_ee_pose(last_drink_poses['place_pre_grasp_pose'])
        self.move_to_joint_positions(self.sim.scene_description.drink_staging_pos)
        self.move_to_joint_positions(self.sim.scene_description.retract_pos)
