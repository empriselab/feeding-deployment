from typing import Any

import numpy as np
import time
import pickle
from scipy.spatial.transform import Rotation
from pathlib import Path
import json
from pybullet_helpers.geometry import Pose

try:
    import rospy
    from std_msgs.msg import Bool
except ModuleNotFoundError:
    ROSPY_IMPORTED = False

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

from feeding_deployment.perception.gestures_perception.static_gesture_detectors import mouth_open, head_nod

from feeding_deployment.actions.feel_the_bite.inside_mouth_transfer import InsideMouthTransfer
from feeding_deployment.actions.feel_the_bite.outside_mouth_transfer import OutsideMouthTransfer

class TransferToolHLA(HighLevelAction):
    """Wipe, or transfer drink, or transfer bite."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tool = None
        self.head_perception_log_dir = self.log_dir / "head_perception_log"
        self.head_perception_log_dir.mkdir(exist_ok=True)

        if self.sim.scene_description.transfer_type == "inside":
            self.transfer = InsideMouthTransfer(self.sim, self.robot_interface, self.perception_interface, self.rviz_interface, self.no_waits, self.head_perception_log_dir)
        elif self.sim.scene_description.transfer_type == "outside":
            self.transfer = OutsideMouthTransfer(self.sim, self.robot_interface, self.perception_interface, self.rviz_interface, self.no_waits, self.head_perception_log_dir)
        else:
            raise ValueError("Bite transfer type not recognized")

        if self.robot_interface is not None:
            self.disable_collision_sensor_pub = rospy.Publisher("/disable_collision_sensor", Bool, queue_size=1)

        self.synthesized_gestures_dict_path = self.gesture_detectors_dir / "synthesized_gestures_dict.json"
        if not self.synthesized_gestures_dict_path.exists():
            with open(self.synthesized_gestures_dict_path, "w") as f:
                f.write("{}")
        
    def set_tool(self, tool):
        self.tool = tool

    def detect_initiate_transfer(self, initiate_transfer_interaction: str, ready_to_initiate_mode: str):
        if initiate_transfer_interaction == "button":
            if self.web_interface is not None:
                self.web_interface.fix_explanation("Please press the button to initiate transfer")
            self.perception_interface.detect_button_press()
        elif initiate_transfer_interaction == "open_mouth":
            if self.web_interface is not None:
                self.web_interface.fix_explanation("Please open your mouth to initiate transfer")
            mouth_open(self.perception_interface, termination_event=None, timeout=600) # 10 minutes
        elif initiate_transfer_interaction == "auto_timeout":
            if self.web_interface is not None:
                self.web_interface.fix_explanation("Please wait for the transfer to initiate in 5 seconds")
            time.sleep(5.0)
        else:
            # Check if the initiate_transfer_interaction is a synthesized gesture.
            gestures = dict(self.load_synthesized_gestures())

            # load from synthesized_gestures_dict_path
            with open(self.synthesized_gestures_dict_path, "r") as f:
                synthesized_gesture_function_name_to_label = json.load(f)

            if initiate_transfer_interaction in gestures:
                self.web_interface.fix_explanation(f"Please do a {synthesized_gesture_function_name_to_label[initiate_transfer_interaction]} to initiate transfer")
                gesture_fn = gestures[initiate_transfer_interaction]
                gesture_fn(self.perception_interface, termination_event=None, timeout=600) # 10 minutes
            else:
                raise NotImplementedError
        print("Initiating transfer")

        if self.web_interface is not None:
            self.web_interface.clear_explanation()

        if ready_to_initiate_mode == "led":
            self.perception_interface.turn_off_led()

    def detect_transfer_complete(self, transfer_complete_interaction: str, ready_for_transfer_interaction: str):
        if transfer_complete_interaction == "button":
            if self.web_interface is not None:
                self.web_interface.fix_explanation("Please press the button to complete transfer")
            self.perception_interface.detect_button_press()
        elif transfer_complete_interaction == "sense":
            if self.tool == "fork":
                if self.web_interface is not None:
                    self.web_interface.fix_explanation("Please bite down on the fork to complete transfer")
                self.perception_interface.detect_force_trigger()
            elif self.tool == "drink":
                if self.web_interface is not None:
                    self.web_interface.fix_explanation("Please do a head nod to complete transfer")
                head_nod(self.perception_interface, termination_event=None, timeout=600) # 10 minutes
            elif self.tool == "wipe":
                if self.web_interface is not None:
                    self.web_interface.fix_explanation("Please do a head nod to complete transfer")
                head_nod(self.perception_interface, termination_event=None, timeout=600) # 10 minutes
        elif transfer_complete_interaction == "auto_timeout":
            if self.web_interface is not None:
                self.web_interface.fix_explanation("Please wait for the transfer to complete in 5 seconds")
            time.sleep(5.0)
        else:
            # Check if the initiate_transfer_interaction is a synthesized gesture.
            gestures = dict(self.load_synthesized_gestures())

            # load from synthesized_gestures_dict_path
            with open(self.synthesized_gestures_dict_path, "r") as f:
                synthesized_gesture_function_name_to_label = json.load(f)
            
            if transfer_complete_interaction in gestures:
                self.web_interface.fix_explanation(f"Please do a {synthesized_gesture_function_name_to_label[transfer_complete_interaction]} to complete transfer")
                gesture_fn = gestures[transfer_complete_interaction]
                gesture_fn(self.perception_interface, termination_event=None, timeout=600) # 10 minutes
            else:
                raise NotImplementedError
        print("Detected transfer completion")

        if self.web_interface is not None:
            self.web_interface.clear_explanation()

        if ready_for_transfer_interaction == "led":
            self.perception_interface.turn_off_led()

    def relay_ready_to_initiate_transfer(self, ready_to_initiate_transfer_interaction: str, initiate_transfer_interaction: str):
        if ready_to_initiate_transfer_interaction == "silent":
            pass
        elif ready_to_initiate_transfer_interaction == "voice":
            if initiate_transfer_interaction == "button":
                self.perception_interface.speak("Please press the button when ready")
            elif initiate_transfer_interaction == "open_mouth":
                self.perception_interface.speak("Please open your mouth when ready")
            elif initiate_transfer_interaction == "auto_timeout":
                self.perception_interface.speak("Please wait 5 seconds for the transfer to initiate")
            else:
                # Check if the initiate_transfer_interaction is a synthesized gesture.
                gestures = dict(self.load_synthesized_gestures())
                # load from synthesized_gestures_dict_path
                with open(self.synthesized_gestures_dict_path, "r") as f:
                    synthesized_gesture_function_name_to_label = json.load(f)

                if initiate_transfer_interaction in gestures:
                    self.perception_interface.speak(f"Please do a {synthesized_gesture_function_name_to_label[initiate_transfer_interaction]} to initiate transfer")
                else:
                    raise NotImplementedError
        elif ready_to_initiate_transfer_interaction == "led":
            self.perception_interface.turn_on_led()
        else:
            raise NotImplementedError

    def relay_ready_for_transfer(self, ready_for_transfer_interaction: str):
        if ready_for_transfer_interaction == "silent":
            pass
        elif ready_for_transfer_interaction == "voice":
            self.perception_interface.speak("Ready for transfer")
        elif ready_for_transfer_interaction == "led":
            self.perception_interface.turn_on_led()
        else:
            raise NotImplementedError

    def execute_transfer(self, ready_to_initiate_mode: str, ready_to_transfer_mode: str,
                         initiate_transfer_mode: str, transfer_complete_mode: str,
                         outside_mouth_distance: float = 0.0,
                         maintain_position_at_goal = False):
        
        self.perception_interface.set_head_perception_tool(self.tool)
        self.perception_interface.start_head_perception_thread()
        if self.robot_interface is not None:
            time.sleep(2.0) # let head perception thread warmstart / robot to stabilize
            self.robot_interface.set_tool(self.tool)
            self.perception_interface.zero_ft_sensor()
        else:
            time.sleep(1.0) # let sim head perception thread warmstart

        if self.sim.scene_description.transfer_type == "inside" and self.robot_interface is not None:
            self.disable_collision_sensor_pub.publish(Bool(data=True))
            print("Sent message to turn off collision sensor")
            time.sleep(0.5) # let collision sensor turn off
            if not self.no_waits:
                input("Press enter to switch to task compliant mode")
            self.robot_interface.switch_to_task_compliant_mode()

        if self.robot_interface is not None:
            self.relay_ready_to_initiate_transfer(ready_to_initiate_mode, initiate_transfer_mode)
            self.detect_initiate_transfer(initiate_transfer_mode, ready_to_initiate_mode)

        self.transfer.set_tool(self.tool)
        self.transfer.move_to_transfer_state(outside_mouth_distance, maintain_position_at_goal)

        if self.robot_interface is not None:
            self.relay_ready_for_transfer(ready_to_transfer_mode)
            self.detect_transfer_complete(transfer_complete_mode, ready_to_transfer_mode)

        # shutdown the head perception thread
        self.perception_interface.stop_head_perception_thread()

        self.transfer.move_to_before_transfer_state()        
        
        if self.sim.scene_description.transfer_type == "inside" and self.robot_interface is not None:                
            if not self.no_waits:
                input("Press enter to switch out of compliant mode")
            self.robot_interface.switch_out_of_compliant_mode()
            self.disable_collision_sensor_pub.publish(Bool(data=False))
            print("Sent message to turn on collision sensor")

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
    
    def transfer_utensil(self, speed: str, *args, **kwargs) -> None:
        assert self.sim.held_object_name == "utensil"

        if self.robot_interface is not None:
            self.robot_interface.set_speed(speed)

        # Assume the last item in args is autocontinue time
        bite_autocontinue_time = args[-1]

        if self.web_interface is not None:
            self.web_interface.set_bite_autocontinue_timeout(bite_autocontinue_time)

        # All other items (everything except the last) should go on to the next call
        remaining_args = args[:-1]

        if self.wrist_interface is not None:
            # start the horizontal spoon thread if it is not already running
            self.wrist_interface.start_horizontal_spoon_thread()

        self.move_to_joint_positions(self.sim.scene_description.before_transfer_pos)

        if self.wrist_interface is not None:
            # stop the keep horizontal thread
            self.wrist_interface.stop_horizontal_spoon_thread()

        self.set_tool("fork")
        self.execute_transfer(*remaining_args, **kwargs)

    def transfer_drink(self, speed: str, *args, **kwargs) -> None:
        assert self.sim.held_object_name == "drink"

        if self.robot_interface is not None:
            self.robot_interface.set_speed(speed)
        
        # Assume the second last item in args is the ask_confirmation
        ask_confirmation = args[-2]

        # Assume the last item in args is autocontinue time
        drink_autocontinue_time = args[-1]

        # All other items (everything except the last two) should go on to the next call
        remaining_args = args[:-2]

        if self.web_interface is not None:
            self.web_interface.set_drink_autocontinue_timeout(drink_autocontinue_time)
            if ask_confirmation:
                self.web_interface.get_drink_transfer_confirmation()

        self.move_to_joint_positions(self.sim.scene_description.before_transfer_pos)

        self.set_tool("drink")    
        self.execute_transfer(*remaining_args, maintain_position_at_goal=True, **kwargs)    

    def transfer_wipe(self, speed: str, *args, **kwargs) -> None:
        assert self.sim.held_object_name == "wipe"

        if self.robot_interface is not None:
            self.robot_interface.set_speed(speed)

        # Assume the last item in args is the ask_confirmation
        ask_confirmation = args[-1]
        # All other items (everything except the last) should go on to the next call
        remaining_args = args[:-1]
        
        self.move_to_joint_positions(self.sim.scene_description.before_transfer_pos)

        if self.web_interface is not None and ask_confirmation:
            self.web_interface.get_wipe_transfer_confirmation()

        self.set_tool("wipe")
        self.execute_transfer(*remaining_args, maintain_position_at_goal=True, **kwargs)
