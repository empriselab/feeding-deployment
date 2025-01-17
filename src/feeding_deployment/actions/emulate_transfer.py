from typing import Any

import numpy as np
import time
import pickle
from scipy.spatial.transform import Rotation
import inspect
from pathlib import Path
import threading

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
from feeding_deployment.actions.feel_the_bite.outside_mouth_transfer import OutsideMouthTransfer
from feeding_deployment.perception.gestures_perception.synthesizer import PersonalizedGestureDetectorSynthesizer
import feeding_deployment.perception.gestures_perception.static_gesture_detectors as static_gesture_detectors

class EmulateTransferHLA(HighLevelAction):
    """Emulate transfer by bringing the empty gripper in front of the user's mouth."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.transfer = OutsideMouthTransfer(self.sim, self.robot_interface, self.perception_interface, self.rviz_interface, self.no_waits)

        self.ready_to_initiate_transfer_interaction = "voice" # "silent", "voice" or "led"
        self.ready_for_transfer_interaction = "voice" # "silent", "voice" or "led"
        self.initiate_transfer_interaction = "open_mouth" # "button", "open_mouth" or "auto_timeout"
        self.transfer_complete_interaction = "button" # "button", "sense" or "auto_timeout"

        self.gesture_examples_path = Path(__file__).parent.parent / "integration" / "log" / "gesture_examples"
        if not self.gesture_examples_path.exists():
            self.gesture_examples_path.mkdir(parents=True)
        self.synthesized_detectors_path = Path(__file__).parent.parent / "perception" / "gestures_perception" / "synthesized_gesture_detectors.py"
        self.detector_synthesizer = PersonalizedGestureDetectorSynthesizer()

        self.test_mode = False

    def detect_initiate_transfer(self):
        if self.initiate_transfer_interaction == "button":
            self.perception_interface.detect_button_press()
        elif self.initiate_transfer_interaction == "open_mouth":
            static_gesture_detectors.mouth_open_detector(self.perception_interface, termination_event=None, timeout=600) # 10 minutes
        elif self.initiate_transfer_interaction == "auto_timeout":
            time.sleep(5.0)
        print("Initiating transfer")

        if self.ready_to_initiate_transfer_interaction == "led":
            self.perception_interface.turn_off_led()

    def detect_transfer_complete(self):
        if self.transfer_complete_interaction == "button":
            self.perception_interface.detect_button_press()
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

    def relay_ready_for_gestures(self):
        if self.ready_for_transfer_interaction == "silent":
            pass
        elif self.ready_for_transfer_interaction == "voice":
            self.perception_interface.speak("Ready for gestures")
        elif self.ready_for_transfer_interaction == "led":
            self.perception_interface.turn_on_led()

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

        if self.web_interface is not None:
            if self.test_mode:
                # find all available gestures
                available_gestures = inspect.getmembers(static_gesture_detectors, inspect.isfunction)

                import feeding_deployment.perception.gestures_perception.synthesized_gesture_detectors as synthesized_gesture_detectors
                available_gestures += inspect.getmembers(synthesized_gesture_detectors, inspect.isfunction)
                gestures_dict = {gesture[0]: gesture[1] for gesture in available_gestures}

                self.web_interface.jump_to_test_gesture_page(list(gestures_dict.keys()))

                # Create a termination event to signal when to switch detectors
                new_gesture_selected_event = threading.Event()
                self.web_interface.start_gesture_listener_thread(new_gesture_selected_event)
                print("Main thread is looking for gestures")

                while True:
                    # Get the selected gesture from the web interface
                    if not new_gesture_selected_event.is_set():
                        time.sleep(0.1)
                        continue
                    selected_gesture = self.web_interface.get_selected_gesture()
                    print(f"Selected gesture: {selected_gesture}")
                    self.web_interface.register_negative_gesture_detection()
                    new_gesture_selected_event.clear()
                    if selected_gesture:
                        # Call the gesture detection function
                        print(f"Calling gesture detection function: {selected_gesture}")
                        gesture_detected = gestures_dict[selected_gesture](self.perception_interface, timeout=600, termination_event=new_gesture_selected_event)
                        if gesture_detected:
                            self.web_interface.register_positive_gesture_detection()
                        else:
                            self.web_interface.register_negative_gesture_detection()
                    else: # None means the user has switched to another page
                        break
                self.web_interface.stop_gesture_listener_thread()

            else:
                # start logging perception data while user selects when to record and delete on the web interface,
                # then extract relevant examples using timestamps
                logging_start_time = self.perception_interface.start_logging_head_perception()
                positive_examples_timestamps, negative_examples_timestamps = self.web_interface.get_gesture_examples()
                self.perception_interface.stop_logging_head_perception()
                
                if len(positive_examples_timestamps) > 0 and len(negative_examples_timestamps) > 0:
                    positive_examples = []
                    for timestamp in positive_examples_timestamps:
                        positive_examples.append(self.perception_interface.extract_from_logged_head_perception_data(timestamp))

                    negative_examples = []
                    for timestamp in negative_examples_timestamps:
                        negative_examples.append(self.perception_interface.extract_from_logged_head_perception_data(timestamp))

                    # save the examples
                    gesture_datapath = self.gesture_examples_path / f"{self.gesture_description}.pkl"
                    with open(gesture_datapath, "wb") as f:
                        pickle.dump({
                            "description": self.gesture_description,
                            "positive_examples": positive_examples, 
                            "negative_examples": negative_examples
                        }, f)

                    input("Press enter to synthesize detector function")
                    generated_function = self.detector_synthesizer.generate_function(gesture_datapath)
                    # Hack to test the synthesizer
                    # hack_datapath = Path(__file__).parent.parent / "perception" / "gestures_perception" / "gestures_examples" / "open_mouth.pkl"
                    # generated_function = self.detector_synthesizer.generate_function(hack_datapath)
                    if generated_function is not None:
                        with open(self.synthesized_detectors_path, "a") as f:
                            f.write(generated_function)
                    else:
                        print("Did not generate valid detector function")
                else:
                    print("Gesture examples recording is not valid")
        else:
            print("Can record or test gestures only with real robot and web interface")
        
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
        print("Params: ", params)
        if params["test_mode"]:
            self.test_mode = True
        else:
            self.gesture_description = params["gesture_description"]
        return super().execute_action(objects, params)