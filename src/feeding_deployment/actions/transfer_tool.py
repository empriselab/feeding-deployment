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

# Rajat ToDo: Move this to a config file
INSIDE_MOUTH_TRANSFER = False 
DISTANCE_INFRONT_MOUTH = 0.20

class TransferToolHLA(HighLevelAction):
    """Wipe, or transfer drink, or transfer bite."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tool = None

        self.ready_for_transfer_interaction = "silent" # "silent", "voice" or "led"
        self.initiate_transfer_interaction = "open_mouth" # "button", "open_mouth" or "auto_timeout"
        self.transfer_complete_interaction = "sense" # "button", "sense" or "auto_timeout"
    
    def set_tool(self, tool):
        self.tool = tool

    def detect_initiate_transfer(self):
        if self.initiate_transfer_interaction == "button":
            self._perception_interface.detect_button_press()
        elif self.initiate_transfer_interaction == "open_mouth":
            self._perception_interface.detect_mouth_open()
        elif self.initiate_transfer_interaction == "auto_timeout":
            self._perception_interface.auto_timeout()
        print("Initiating transfer")

    def detect_transfer_complete(self):
        if self.transfer_complete_interaction == "button":
            self._perception_interface.detect_button_press()
        elif self.transfer_complete_interaction == "sense":
            if self.tool == "fork":
                self._perception_interface.detect_force_trigger()
            elif self.tool == "drink":
                self._perception_interface.detect_head_shake()
            elif self.tool == "wipe":
                self._perception_interface.detect_head_still()
        elif self.transfer_complete_interaction == "auto_timeout":
            self._perception_interface.auto_timeout()
        print("Detected transfer completion")

    def relay_ready_for_transfer(self):
        if self.ready_for_transfer_interaction == "silent":
            pass
        elif self.ready_for_transfer_interaction == "voice":
            self._perception_interface.speak("Please ooopen your mouth when ready")

    def execute_transfer_loop(self, maintain_position_at_goal = False):
        
        assert self._perception_interface.head_perception_thread_is_running(), "Head perception thread is not running"
        assert self.tool is not None, "Tool is not set"

        self.relay_ready_for_transfer()
        self.detect_initiate_transfer()

        print("Setting state to 1")

        # move to infront of mouth
        forque_target_base = self._perception_interface.get_head_perception_tool_tip_target_pose()
        servo_point_forque_target = np.identity(4)
        servo_point_forque_target[:3,3] = np.array([0, 0, -DISTANCE_INFRONT_MOUTH]).reshape(1,3)
        infront_mouth_target = forque_target_base @ servo_point_forque_target

        # mouth is assumed to be facing away from the wheelchair
        infront_mouth_target[:3, :3] = Rotation.from_quat([0.478, -0.505, -0.515, 0.502]).as_matrix()
        if self.tool == "fork":
            wrist_to_tip = self._sim.scene_description.tool_frame_to_utensil_tip
        elif self.tool == "drink":
            wrist_to_tip = self._sim.scene_description.tool_frame_to_drink_tip
        elif self.tool == "wipe":
            wrist_to_tip = self._sim.scene_description.tool_frame_to_wipe_tip
        else:
            raise ValueError("Tool not recognized")
        
        tip_to_wrist = np.linalg.inv(wrist_to_tip.to_matrix())
        tool_frame_target = infront_mouth_target @ tip_to_wrist

        target_pose = Pose.from_matrix(tool_frame_target)

        self.move_to_ee_pose(target_pose)

        self.detect_transfer_complete()
        # shutdown the head perception thread
        self._perception_interface.stop_head_perception_thread()

        self.move_to_ee_pose(self._sim.scene_description.before_transfer_pose)

        # incase for some reason the head perception thread is still running
        self._perception_interface.stop_head_perception_thread()            
        print("Exiting transfer loop")

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

    def execute_action(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> None:
        assert len(objects) == 1
        tool = objects[0]

        if tool.name == "utensil":
            assert self._sim.held_object_name == "utensil"

            self.move_to_joint_positions(self._sim.scene_description.before_transfer_pos)

            if self.wrist_controller is not None:
                # stop the keep horizontal thread
                self.wrist_controller.stop_horizontal_spoon_thread()

            self._perception_interface.set_head_perception_tool("fork")
            self._perception_interface.start_head_perception_thread()
            if self._robot_interface is not None:
                time.sleep(5.0) # let head perception thread warmstart / robot to stabilize
                self._robot_interface.set_tool("fork")
            else:
                time.sleep(1.0) # let sim head perception thread warmstart
            self.set_tool("fork")

            if INSIDE_MOUTH_TRANSFER and self._robot_interface is not None:
                if not self.no_waits:
                    input("Press enter to switch to task compliant mode")
                self._robot_interface.switch_to_task_compliant_mode()
                
            self.execute_transfer_loop()

            if INSIDE_MOUTH_TRANSFER and self._robot_interface is not None:                
                if not self.no_waits:
                    input("Press enter to switch out of compliant mode")
                self._robot_interface.switch_out_of_compliant_mode()

            # Send message to web interface indicating transfer is done.
            if self._web_interface is not None:
                self._web_interface.send_web_interface_message({"state": "bite_transfer", "status": "completed"})
        
        elif tool.name == "drink":

            assert self._sim.held_object_name == "drink"
            
            self.move_to_joint_positions(self._sim.scene_description.before_transfer_pos)

            self._perception_interface.set_head_perception_tool("drink")
            self._perception_interface.start_head_perception_thread()
            if self._robot_interface is not None:
                time.sleep(5.0) # let head perception thread warmstart / robot to stabilize
                self._robot_interface.set_tool("drink")
            else:
                time.sleep(1.0) # let sim head perception thread warmstart
            self.set_tool("drink")

            if INSIDE_MOUTH_TRANSFER and self._robot_interface is not None:
                if not self.no_waits:
                    input("Press enter to switch to task compliant mode")
                self._robot_interface.switch_to_task_compliant_mode()
                
            self.execute_transfer_loop(maintain_position_at_goal=True)

            if INSIDE_MOUTH_TRANSFER and self._robot_interface is not None:                
                if not self.no_waits:
                    input("Press enter to switch out of compliant mode")
                self._robot_interface.switch_out_of_compliant_mode()

            # Send message to web interface indicating transfer is done.
            if self._web_interface is not None:
                self._web_interface.send_web_interface_message({"state": "drink_transfer", "status": "completed"})
        
        elif tool.name == "wipe":

            assert self._sim.held_object_name == "wipe"
            
            self.move_to_joint_positions(self._sim.scene_description.before_transfer_pos)

            self._perception_interface.set_head_perception_tool("wipe")
            self._perception_interface.start_head_perception_thread()
            if self._robot_interface is not None:
                time.sleep(5.0) # let head perception thread warmstart / robot to stabilize
                self._robot_interface.set_tool("wipe")
            else:
                time.sleep(1.0) # let sim head perception thread warmstart
            self.set_tool("wipe")

            if INSIDE_MOUTH_TRANSFER and self._robot_interface is not None:
                if not self.no_waits:
                    input("Press enter to switch to task compliant mode")
                self._robot_interface.switch_to_task_compliant_mode()
                
            self.execute_transfer_loop(maintain_position_at_goal=True)

            if INSIDE_MOUTH_TRANSFER and self._robot_interface is not None:                
                if not self.no_waits:
                    input("Press enter to switch out of compliant mode")
                self._robot_interface.switch_out_of_compliant_mode()

            # Send message to web interface indicating transfer is done.
            if self._web_interface is not None:
                self._web_interface.send_web_interface_message({"state": "moved_to_wiping_position", "status": "completed"})

        else:
            print(f"TransferTool not yet implemented for {tool}")
            return []
