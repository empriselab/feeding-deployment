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

class PickToolHLA(HighLevelAction):
    """Pick up a tool (utensil, drink, or wipe)."""

    def get_name(self) -> str:
        return "PickTool"

    def get_operator(self) -> LiftedOperator:
        tool = Variable("?tool", tool_type)
        return LiftedOperator(
            self.get_name(),
            parameters=[tool],
            preconditions={LiftedAtom(GripperFree, [])},
            add_effects={Holding([tool])},
            delete_effects={LiftedAtom(GripperFree, [])},
        )
    
    def get_behavior_tree(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> BehaviorTreeNode:
        del params  # not used right now
        
        assert len(objects) == 1
        tool = objects[0]
        assert self.sim.held_object_name is None

        if tool.name == "utensil" and self.sim.scene_description.scene_label == "vention":
            yaml_filename = "pick_utensil.yaml"
        elif tool.name == "drink":
            yaml_filename = "pick_drink.yaml"
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

            assert self.sim.held_object_name is None
            
            self.move_to_joint_positions(self.sim.scene_description.retract_pos)
            self.close_gripper()

            if self.sim.scene_description.scene_label == "vention":
                self.move_to_joint_positions(self.sim.scene_description.wipe_infront_mount_pos)

            self.move_to_joint_positions(self.sim.scene_description.wipe_above_mount_pos)
            self.move_to_ee_pose(self.sim.scene_description.wipe_inside_mount)
            self.grasp_tool("wipe")
            self.move_to_ee_pose(self.sim.scene_description.wipe_outside_mount)
            
            if self.sim.scene_description.scene_label == "wheelchair":
                self.move_to_ee_pose(self.sim.scene_description.wipe_outside_above_mount)
            elif self.sim.scene_description.scene_label == "vention":
                self.move_to_joint_positions(self.sim.scene_description.wipe_neutral_pos)
            self.move_to_joint_positions(self.sim.scene_description.retract_pos)
            self.move_to_joint_positions(self.sim.scene_description.before_transfer_pos)

            # Send message to web interface.
            if self.web_interface is not None:
                self.web_interface.send_web_interface_message({"state": "prepare_mouth_wiping", "status": "completed"})

        else:
            print(f"PickTool not yet implemented for {tool}")
            return []
        
    def pick_drink(self) -> None:
        assert self.sim.held_object_name is None

        self.move_to_joint_positions(self.sim.scene_description.retract_pos)
        self.close_gripper()
        self.move_to_joint_positions(self.sim.scene_description.drink_gaze_pos)

        drink_poses = self.perception_interface.perceive_drink_pickup_poses()

        self.move_to_joint_positions(self.sim.scene_description.drink_staging_pos)
        self.move_to_ee_pose(drink_poses['pre_grasp_pose'])
        self.move_to_ee_pose(drink_poses['inside_bottom_pose'])
        self.move_to_ee_pose(drink_poses['inside_top_pose'])
        self.grasp_tool("drink")
        self.move_to_ee_pose(drink_poses['post_grasp_pose'])

        self.perception_interface.record_drink_pickup_joint_pos()

        self.move_to_joint_positions(self.sim.scene_description.before_transfer_pos)

        # Send message to web interface.
        if self.web_interface is not None:
            self.web_interface.send_web_interface_message({"state": "drink_pickup", "status": "completed"})
