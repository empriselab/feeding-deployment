from typing import Any

import time
import py_trees

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
    FunctionalSingletonBTBehaviour,
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

            # Create behavior tree.
            assert self.sim.held_object_name is None
            
            move_to_retract = FunctionalSingletonBTBehaviour(
                "MoveToRetract",
                lambda: self.move_to_joint_positions(self.sim.scene_description.retract_pos)
            )
            close_gripper = FunctionalSingletonBTBehaviour(
                "CloseGripper",
                lambda: self.close_gripper()
            )
            move_to_above_mount = FunctionalSingletonBTBehaviour(
                "MoveToAboveMount",
                lambda: self.move_to_joint_positions(self.sim.scene_description.utensil_above_mount_pos)
            )
            move_to_utensil_inside_mount = FunctionalSingletonBTBehaviour(
                "MoveToUtensilInsideMount",
                lambda: self.move_to_ee_pose(self.sim.scene_description.utensil_inside_mount)
            )
            grasp_utensil = FunctionalSingletonBTBehaviour(
                "GraspUtensil",
                lambda: self.grasp_tool("utensil")
            )

            # TODO: should we put this logic in a condition node?
            def _reset_wrist():
                if self.wrist_interface is not None:
                    time.sleep(1.0) # wait for the utensil to be connected
                    print("Resetting wrist controller ...")
                    self.wrist_interface.set_velocity_mode()
                    self.wrist_interface.reset()

            reset_wrist = FunctionalSingletonBTBehaviour(
                "ResetWrist",
                _reset_wrist,
            )

            move_to_utensil_outside_mount = FunctionalSingletonBTBehaviour(
                "MoveToUtensilOutsideMount",
                lambda: self.move_to_ee_pose(self.sim.scene_description.utensil_outside_mount)
            )

            # TODO: should we put this logic in a condition node?
            def _move_to_utensil_outside_above_mount():
                if self.sim.scene_description.scene_label == "vention":
                    self.move_to_ee_pose(self.sim.scene_description.utensil_outside_above_mount)

            move_to_utensil_outside_above_mount = FunctionalSingletonBTBehaviour(
                "MoveToUtensilOutsideAboveMount",
                _move_to_utensil_outside_above_mount,
            )

            # As far as I can tell, we need to create a separate node, even though
            # this is the same as the retract above.
            move_to_retract2 = FunctionalSingletonBTBehaviour(
                "MoveToRetract",
                lambda: self.move_to_joint_positions(self.sim.scene_description.retract_pos)
            )

            # Pre-emptively move to the before_transfer_pos because moving to above_plate_pos from retract_pos is unsafe.
            move_to_before_transfer_pose = FunctionalSingletonBTBehaviour(
                "MoveToBeforeTransferPos",
                lambda: self.move_to_joint_positions(self.sim.scene_description.before_transfer_pos)
            )

            # Finalize the behaviour tree.
            behaviour_sequence = [
                move_to_retract,
                close_gripper,
                move_to_above_mount,
                move_to_utensil_inside_mount,
                grasp_utensil,
                reset_wrist,
                move_to_utensil_outside_mount,
                move_to_utensil_outside_above_mount,
                move_to_retract2,
                move_to_before_transfer_pose,
            ]

            root = py_trees.composites.Sequence(name="Root", memory=True)
            root.add_children(behaviour_sequence)

            # Execute the behavior tree.
            print("Executing behaviour tree...")
            root.setup_with_descendants()
            root.tick_once()
            print(py_trees.display.unicode_tree(root=root, show_status=True))
            assert root.status == py_trees.common.Status.SUCCESS
            
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