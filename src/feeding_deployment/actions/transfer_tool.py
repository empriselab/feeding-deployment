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
    tool_type,
    GripperFree,
    Holding,
    ToolPrepared,
    ToolTransferDone,
)

from feeding_deployment.perception.gestures_perception.static_gesture_detectors import mouth_open_detector, head_shake_detector, head_still_detector

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

        self.ready_to_initiate_transfer_interaction = "led" # "silent", "voice" or "led"
        self.ready_for_transfer_interaction = "led" # "silent", "voice" or "led"
        self.initiate_transfer_interaction = "open_mouth" # "button", "open_mouth" or "auto_timeout"
        self.transfer_complete_interaction = "sense" # "button", "sense" or "auto_timeout"
    
    def set_tool(self, tool):
        self.tool = tool

    def detect_initiate_transfer(self):
        if self.initiate_transfer_interaction == "button":
            self.perception_interface.detect_button_press()
        elif self.initiate_transfer_interaction == "open_mouth":
            mouth_open_detector(self.perception_interface, timeout=600) # 10 minutes
        elif self.initiate_transfer_interaction == "auto_timeout":
            time.sleep(5.0)
        print("Initiating transfer")

        if self.ready_to_initiate_transfer_interaction == "led":
            self.perception_interface.turn_off_led()

    def detect_transfer_complete(self):
        if self.transfer_complete_interaction == "button":
            self.perception_interface.detect_button_press()
        elif self.transfer_complete_interaction == "sense":
            if self.tool == "fork":
                self.perception_interface.detect_force_trigger()
            elif self.tool == "drink":
                head_shake_detector(self.perception_interface, timeout=600) # 10 minutes
            elif self.tool == "wipe":
                head_still_detector(self.perception_interface, timeout=600) # 10 minutes
        elif self.transfer_complete_interaction == "auto_timeout":
            time.sleep(5.0)
        print("Detected transfer completion")

        if self.ready_for_transfer_interaction == "led":
            self.perception_interface.turn_off_led()

    def relay_ready_to_initiate_transfer(self):
        if self.ready_to_initiate_transfer_interaction == "silent":
            pass
        elif self.ready_to_initiate_transfer_interaction == "voice":
            self.perception_interface.speak("Please open your mouth when ready")
        elif self.ready_to_initiate_transfer_interaction == "led":
            self.perception_interface.turn_on_led()

    def relay_ready_for_transfer(self):
        if self.ready_for_transfer_interaction == "silent":
            pass
        elif self.ready_for_transfer_interaction == "voice":
            self.perception_interface.speak("Ready for transfer")
        elif self.ready_for_transfer_interaction == "led":
            self.perception_interface.turn_on_led()

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

        if self.robot_interface is not None:
            self.relay_ready_to_initiate_transfer()
            self.detect_initiate_transfer()

        self.transfer.set_tool(self.tool)
        self.transfer.move_to_transfer_state(maintain_position_at_goal)

        if self.robot_interface is not None:
            self.relay_ready_for_transfer()
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
            delete_effects={ToolPrepared([tool])},
        )
    
    def get_behavior_tree_filename(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> str:
        del params  # not used right now
        assert len(objects) == 1
        tool = objects[0]
        assert tool.name in ["utensil", "drink", "wipe"]
        return f"transfer_{tool.name}.yaml"    
    
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

    def transfer_drink(self) -> None:
        assert self.sim.held_object_name == "drink"
        
        self.move_to_joint_positions(self.sim.scene_description.before_transfer_pos)

        self.set_tool("drink")    
        self.execute_transfer(maintain_position_at_goal=True)

        # Send message to web interface indicating transfer is done.
        if self.web_interface is not None:
            self.web_interface.send_web_interface_message({"state": "drink_transfer", "status": "completed"})        

    def transfer_wipe(self) -> None:
        assert self.sim.held_object_name == "wipe"
        
        self.move_to_joint_positions(self.sim.scene_description.before_transfer_pos)

        self.set_tool("wipe")
        self.execute_transfer(maintain_position_at_goal=True)

        # Send message to web interface indicating transfer is done.
        if self.web_interface is not None:
            self.web_interface.send_web_interface_message({"state": "moved_to_wiping_position", "status": "completed"})
