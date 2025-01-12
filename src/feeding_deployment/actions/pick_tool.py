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
    
    def get_behavior_tree_filename(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> str:
        del params  # not used right now
        assert len(objects) == 1
        tool = objects[0]
        assert self.sim.scene_description.scene_label == "vention"
        assert tool.name in ["utensil", "drink", "wipe"]
        return f"pick_{tool.name}.yaml"
    
    def pick_utensil(self) -> None:
        assert self.sim.held_object_name is None
            
        self.move_to_joint_positions(self.sim.scene_description.retract_pos)
        self.close_gripper()
        self.move_to_joint_positions(self.sim.scene_description.utensil_above_mount_pos)
        self.move_to_ee_pose(self.sim.scene_description.utensil_inside_mount)
        self.grasp_tool("utensil")

        if self.wrist_interface is not None:
            time.sleep(1.0) # wait for the utensil to be connected
            print("Resetting wrist controller ...")
            self.wrist_interface.set_velocity_mode()
            self.wrist_interface.reset()

        self.move_to_ee_pose(self.sim.scene_description.utensil_outside_mount)
        if self.sim.scene_description.scene_label == "vention":
            self.move_to_ee_pose(self.sim.scene_description.utensil_outside_above_mount)
        elif self.sim.scene_description.scene_label == "wheelchair":
            # Not sure if this is necessary.
            self.move_to_joint_positions(self.sim.scene_description.retract_pos)
        # Pre-emptively move to the before_transfer_pos because moving to above_plate_pos from retract_pos is unsafe.
        self.move_to_joint_positions(self.sim.scene_description.before_transfer_pos)
        
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

    def pick_wipe(self) -> None:
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
