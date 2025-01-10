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
    EmulateTransferDone,
)
from feeding_deployment.perception.gestures_perception.static_gesture_detectors import mouth_open_detector
from feeding_deployment.actions.feel_the_bite.outside_mouth_transfer import OutsideMouthTransfer

class EmulateTransferHLA(HighLevelAction):
    """Emulate transfer by bringing the empty gripper in front of the user's mouth."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.transfer = OutsideMouthTransfer(self.sim, self.robot_interface, self.perception_interface, self.rviz_interface, self.no_waits)

        self.ready_for_transfer_interaction = "voice" # "silent", "voice" or "led"
        self.initiate_transfer_interaction = "open_mouth" # "button", "open_mouth" or "auto_timeout"
        self.transfer_complete_interaction = "button" # "button", "sense" or "auto_timeout"

    def detect_initiate_transfer(self):
        if self.initiate_transfer_interaction == "button":
            self.perception_interface.detect_button_press()
        elif self.initiate_transfer_interaction == "open_mouth":
            mouth_open_detector(self.perception_interface, timeout=600) # 10 minutes
        elif self.initiate_transfer_interaction == "auto_timeout":
            time.sleep(5.0)
        print("Initiating transfer")

    def detect_transfer_complete(self):
        if self.transfer_complete_interaction == "button":
            self.perception_interface.detect_button_press()
        elif self.transfer_complete_interaction == "auto_timeout":
            time.sleep(5.0)
        print("Detected transfer completion")

    def relay_ready_to_initiate_transfer(self):
        if self.ready_for_transfer_interaction == "silent":
            pass
        elif self.ready_for_transfer_interaction == "voice":
            self.perception_interface.speak("Please open your mouth when ready")

    def relay_ready_for_gestures(self):
        if self.ready_for_transfer_interaction == "silent":
            pass
        elif self.ready_for_transfer_interaction == "voice":
            self.perception_interface.speak("Ready for gestures")

    def emulate_transfer(self):

        self.move_to_joint_positions(self.sim.scene_description.before_transfer_pos)

        self.perception_interface.set_head_perception_tool("fork")
        self.perception_interface.start_head_perception_thread()
        if self.robot_interface is not None:
            time.sleep(5.0) # let head perception thread warmstart / robot to stabilize
            self.robot_interface.set_tool("fork")
            self.perception_interface.zero_ft_sensor()
        else:
            time.sleep(1.0) # let sim head perception thread warmstart

        if self.robot_interface is not None:
            self.relay_ready_to_initiate_transfer()
            self.detect_initiate_transfer()

        self.transfer.set_tool("fork")
        self.transfer.move_to_transfer_state()

        if self.robot_interface is not None:
            self.relay_ready_for_gestures()

        if self.test_mode:
            # Test new gestures at this state using the web application
            pass
        else:
            # Record new gestures at this state using the web application
            pass
        
        if self.robot_interface is not None:
            self.detect_transfer_complete()

        # shutdown the head perception thread
        self.perception_interface.stop_head_perception_thread()

        self.transfer.move_to_before_transfer_state()        

    def get_name(self) -> str:
        return "EmulateTransfer"
    
    def get_operator(self) -> LiftedOperator:
        return LiftedOperator(
            self.get_name(),
            parameters=[],
            preconditions={LiftedAtom(GripperFree, [])},
            add_effects={LiftedAtom(EmulateTransferDone, [])},
            delete_effects=set(),
        )
    
    def get_behavior_tree_filename(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> str:
        return f"emulate_transfer.yaml"
    
    def execute_action(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> None:
        if params["test_mode"]:
            self.test_mode = True
        return super().execute_action(objects, params)