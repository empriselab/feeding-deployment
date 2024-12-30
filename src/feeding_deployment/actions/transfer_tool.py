from typing import Any

import numpy as np
import time
import pickle
from scipy.spatial.transform import Rotation

from pybullet_helpers.geometry import Pose

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
    ToolPrepared,
    ToolTransferDone,
)

from feeding_deployment.actions.feel_the_bite.inside_mouth_transfer import InsideMouthTransfer
from feeding_deployment.actions.feel_the_bite.outside_mouth_transfer import OutsideMouthTransfer

class TransferToolHLA(HighLevelAction):
    """Wipe, or transfer drink, or transfer bite."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tool = None

        if self.sim.scene_description.transfer_type == "inside":
            self.transfer = InsideMouthTransfer(self.sim, self.robot_interface, self.perception_interface, self.rviz_interface, self.no_waits)
        elif self.sim.scene_description.transfer_type == "outside":
            self.transfer = OutsideMouthTransfer(self.sim, self.robot_interface, self.perception_interface, self.rviz_interface, self.no_waits)
        else:
            raise ValueError("Bite transfer type not recognized")

        self.ready_for_transfer_interaction = "silent" # "silent", "voice" or "led"
        self.initiate_transfer_interaction = "open_mouth" # "button", "open_mouth" or "auto_timeout"
        self.transfer_complete_interaction = "sense" # "button", "sense" or "auto_timeout"
    
    def set_tool(self, tool):
        self.tool = tool

    def detect_initiate_transfer(self):
        if self.initiate_transfer_interaction == "button":
            self.perception_interface.detect_button_press()
        elif self.initiate_transfer_interaction == "open_mouth":
            self.perception_interface.detect_mouth_open()
        elif self.initiate_transfer_interaction == "auto_timeout":
            self.perception_interface.auto_timeout()
        print("Initiating transfer")

    def detect_transfer_complete(self):
        if self.transfer_complete_interaction == "button":
            self.perception_interface.detect_button_press()
        elif self.transfer_complete_interaction == "sense":
            if self.tool == "fork":
                self.perception_interface.detect_force_trigger()
            elif self.tool == "drink":
                self.perception_interface.detect_head_shake()
            elif self.tool == "wipe":
                self.perception_interface.detect_head_still()
        elif self.transfer_complete_interaction == "auto_timeout":
            self.perception_interface.auto_timeout()
        print("Detected transfer completion")

    def relay_ready_for_transfer(self):
        if self.ready_for_transfer_interaction == "silent":
            pass
        elif self.ready_for_transfer_interaction == "voice":
            self.perception_interface.speak("Please ooopen your mouth when ready")

    def execute_transfer(self, maintain_position_at_goal = False):

        self.perception_interface.set_head_perception_tool(self.tool)
        self.perception_interface.start_head_perception_thread()
        if self.robot_interface is not None:
            time.sleep(5.0) # let head perception thread warmstart / robot to stabilize
            self.robot_interface.set_tool(self.tool)
            self.perception_interface.zero_ft_sensor()
        else:
            time.sleep(1.0) # let sim head perception thread warmstart

        if self.sim.scene_description.transfer_type == "inside" and self.robot_interface is not None:
            if not self.no_waits:
                input("Press enter to switch to task compliant mode")
            self.robot_interface.switch_to_task_compliant_mode()

        self.relay_ready_for_transfer()
        self.detect_initiate_transfer()

        self.transfer.set_tool(self.tool)
        self.transfer.move_to_transfer_state(maintain_position_at_goal)

        self.detect_transfer_complete()

        # shutdown the head perception thread
        self.perception_interface.stop_head_perception_thread()

        self.transfer.move_to_before_transfer_state()        
        
        if self.sim.scene_description.transfer_type == "inside" and self.robot_interface is not None:                
            if not self.no_waits:
                input("Press enter to switch out of compliant mode")
            self.robot_interface.switch_out_of_compliant_mode()

    def get_name(self) -> str:
        return "TransferTool"

    def get_operator(self) -> LiftedOperator:
        tool = Variable("?tool", tool_type)
        return LiftedOperator(
            self.get_name(),
            parameters=[tool],
            preconditions={Holding([tool]), ToolPrepared([tool])},
            add_effects={LiftedAtom(ToolTransferDone, [tool])},
            delete_effects=set(),
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
            yaml_filename = "transfer_utensil.yaml"
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

        if tool.name == "utensil":
            # Get and execute the behavior tree.
            behavior_tree = self.get_behavior_tree(objects, params)
            behavior_tree.tick()
        
        elif tool.name == "drink":

            assert self.sim.held_object_name == "drink"
            
            self.move_to_joint_positions(self.sim.scene_description.before_transfer_pos)

            self.set_tool("drink")    
            self.execute_transfer(maintain_position_at_goal=True)

            # Send message to web interface indicating transfer is done.
            if self.web_interface is not None:
                self.web_interface.send_web_interface_message({"state": "drink_transfer", "status": "completed"})
        
        elif tool.name == "wipe":

            assert self.sim.held_object_name == "wipe"
            
            self.move_to_joint_positions(self.sim.scene_description.before_transfer_pos)

            self.set_tool("wipe")
            self.execute_transfer(maintain_position_at_goal=True)

            # Send message to web interface indicating transfer is done.
            if self.web_interface is not None:
                self.web_interface.send_web_interface_message({"state": "moved_to_wiping_position", "status": "completed"})

        else:
            print(f"TransferTool not yet implemented for {tool}")
            return []

    def transfer_utensil(self) -> None:
        assert self.sim.held_object_name == "utensil"

        if self.wrist_interface is not None:
            # start the horizontal spoon thread if it is not already running
            self.wrist_interface.start_horizontal_spoon_thread()

        self.move_to_joint_positions(self.sim.scene_description.before_transfer_pos)

        if self.wrist_interface is not None:
            # stop the keep horizontal thread
            self.wrist_interface.stop_horizontal_spoon_thread()

        self.set_tool("fork")
        self.execute_transfer()

        # Send message to web interface indicating transfer is done.
        if self.web_interface is not None:
            self.web_interface.send_web_interface_message({"state": "bite_transfer", "status": "completed"})
