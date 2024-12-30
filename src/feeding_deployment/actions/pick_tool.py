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
    parse_behavior_tree,
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
        
        assert len(objects) == 1
        tool = objects[0]
        assert tool.name == "utensil"  # other things forthcoming
        del params  # not used right now

        assert self.sim.held_object_name is None

        # Need to figure out how to handle conditional statements...
        assert self.sim.scene_description.scene_label == "vention"

        # We'll move this into a file later.
        yaml_str = """
name: "Root"
type: "Sequence"
children:

  - name: "MoveToRetractPos"
    type: "Behavior"
    parameters:
      - name: "RetractPosition"
        space:
          type: "Box"
          lower: !hla arm_joint_lower_limits
          upper: !hla arm_joint_upper_limits
        is_user_editable: False
        value: !scene_description retract_pos
    fn: !hla move_to_joint_positions

  - name: "CloseGripper"
    type: "Behavior"
    parameters: []
    fn: !hla close_gripper

  - name: "MoveToUtensilAboveMountPos"
    type: "Behavior"
    parameters:
      - name: "UtensilAboveMountPos"
        space:
          type: "Box"
          lower: !hla arm_joint_lower_limits
          upper: !hla arm_joint_upper_limits
        is_user_editable: False
        value: !scene_description utensil_above_mount_pos
    fn: !hla move_to_joint_positions

  - name: "MoveUtensilInsideMount"
    type: "Behavior"
    parameters:
      - name: "UtensilInsideMountEE"
        space:
          type: "PoseSpace"
        is_user_editable: False
        value: !scene_description utensil_inside_mount
    fn: !hla move_to_ee_pose

  - name: "GraspUtensil"
    type: "Behavior"
    parameters:
      - name: "Tool"
        space:
          type: "Enum"
          elements: ["utensil", "drink", "wipe"]
        is_user_editable: False
        value: "utensil"
    fn: !hla grasp_tool

  - name: "MoveUtensilOutsideMount"
    type: "Behavior"
    parameters:
      - name: "MoveUtensilOutsideMountEE"
        space:
          type: "PoseSpace"
        is_user_editable: False
        value: !scene_description utensil_outside_mount
    fn: !hla move_to_ee_pose

  # Assuming here self.sim.scene_description.scene_label == "vention"
  - name: "MoveUtensilOutsideAboveMount"
    type: "Behavior"
    parameters:
      - name: "MoveUtensilOutsideAboveMountEE"
        space:
          type: "PoseSpace"
        is_user_editable: False
        value: !scene_description utensil_outside_above_mount
    fn: !hla move_to_ee_pose

  - name: "MoveToRetractPos"
    type: "Behavior"
    parameters:
      - name: "RetractPosition"
        space:
          type: "Box"
          lower: !hla arm_joint_lower_limits
          upper: !hla arm_joint_upper_limits
        is_user_editable: False
        value: !scene_description retract_pos
    fn: !hla move_to_joint_positions

  # Pre-emptively move to the before_transfer_pos because moving to above_plate_pos
  # from retract_pos is unsafe.
  - name: "MoveToBeforeTransferPos"
    type: "Behavior"
    parameters:
      - name: "BeforeTransferPos"
        space:
          type: "Box"
          lower: !hla arm_joint_lower_limits
          upper: !hla arm_joint_upper_limits
        is_user_editable: False
        value: !scene_description before_transfer_pos
    fn: !hla move_to_joint_positions
"""
        bt = parse_behavior_tree(yaml_str, self)
        return bt

    def execute_action(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> None:
        assert len(objects) == 1
        tool = objects[0]

        if tool.name == "drink":

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

        elif tool.name == "utensil":

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